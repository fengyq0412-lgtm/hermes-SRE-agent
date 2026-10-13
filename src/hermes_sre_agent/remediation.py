"""V0.2 本机运维变更：白名单计划、一次性审批和写回后验证。"""

import hashlib
import fcntl
import json
import os
import re
import secrets
import sqlite3
import stat
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4


TOKEN_TTL_SECONDS = 600
MAX_CONFIG_BYTES = 64_000
CONTAINER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class _NoHealthRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


_HEALTH_OPENER = build_opener(_NoHealthRedirect())


class RemediationError(ValueError):
    """计划、审批或执行条件不满足。"""


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_object(data):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RemediationError(f"JSON 包含重复字段：{key}")
            result[key] = value
        return result

    try:
        value = json.loads(data, object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemediationError("配置或策略不是有效的 UTF-8 JSON。") from exc
    if not isinstance(value, dict):
        raise RemediationError("配置或策略必须是 JSON 对象。")
    return value


def _read_regular(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RemediationError(f"只接受普通且非硬链接的文件：{path}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES:
        raise RemediationError(f"文件超过 {MAX_CONFIG_BYTES} 字节：{path}")
    return data


def _atomic_replace(path, content, mode, expected_sha):
    path = Path(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary = ".hermes-change-" + uuid4().hex
    try:
        if _sha(_read_regular(path)) != expected_sha:
            raise RemediationError("配置在写入前发生变化，计划失效。")
        target = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         mode, dir_fd=fd)
        with os.fdopen(target, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), mode)
        os.replace(temporary, path.name, src_dir_fd=fd, dst_dir_fd=fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=fd)
        except FileNotFoundError:
            pass
        os.close(fd)


class LocalDockerAdapter:
    """固定 Docker 参数与本机 HTTP 探针，不接受任意命令。"""

    def _run(self, arguments, timeout=20):
        try:
            result = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RemediationError(f"Docker 操作未完成：{exc}") from exc
        if result.returncode:
            raise RemediationError("Docker 操作失败：" + (result.stderr.strip() or result.stdout.strip())[:240])
        return result.stdout.strip()

    def container_id(self, name):
        if not CONTAINER_NAME.fullmatch(name):
            raise RemediationError("容器名不在允许的格式范围内。")
        result = self._run(["docker", "inspect", "--type", "container", "--format", "{{.Id}}", name])
        if not re.fullmatch(r"[0-9a-f]{64}", result):
            raise RemediationError("无法确认容器身份。")
        return result

    def restart(self, name, expected_id):
        if self.container_id(name) != expected_id:
            raise RemediationError("目标容器已替换，请重新生成变更计划。")
        self._run(["docker", "restart", "--time", "10", name], timeout=35)

    def health(self, url):
        started = time.monotonic()
        try:
            with _HEALTH_OPENER.open(Request(url, method="GET"), timeout=8) as response:
                status, body = response.status, response.read(16_385)
        except HTTPError as exc:
            status, body = exc.code, exc.read(16_385)
        except (URLError, TimeoutError, OSError) as exc:
            raise RemediationError(f"健康检查不可达：{exc}") from exc
        if len(body) > 16_384:
            raise RemediationError("健康检查响应过大。")
        try:
            payload = _json_object(body)
        except RemediationError:
            payload = {}
        return {"status_code": status, "latency_ms": round((time.monotonic() - started) * 1000, 1),
                "body": payload}


class ChangeManager:
    """审批令牌仅以摘要保存，执行前原子消费；计划绑定策略、配置和容器版本。"""

    def __init__(self, policy_path, state_dir, adapter=None):
        self.policy_path = Path(policy_path).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.adapter = adapter or LocalDockerAdapter()
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.state_dir, 0o700)
        self.db_path = self.state_dir / "changes.sqlite3"
        with sqlite3.connect(self.db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS changes (id TEXT PRIMARY KEY, plan TEXT NOT NULL, "
                       "status TEXT NOT NULL, token_hash TEXT, token_expires REAL, result TEXT)")
        os.chmod(self.db_path, 0o600)

    def _service(self, name):
        policy_bytes = _read_regular(self.policy_path)
        policy = _json_object(policy_bytes)
        services = policy.get("services")
        if policy.get("version") != 1 or not isinstance(services, dict) or name not in services:
            raise RemediationError("服务不在 v1 变更策略白名单中。")
        service = services[name]
        if not isinstance(service, dict):
            raise RemediationError("服务策略格式无效。")
        relative = service.get("config_file")
        if (not isinstance(relative, str) or not relative.endswith(".json")
                or Path(relative).is_absolute() or any(part in {"..", "."} or part.startswith(".")
                                                  for part in Path(relative).parts)):
            raise RemediationError("配置路径必须是策略目录下的非隐藏 JSON 文件。")
        config_path = self.policy_path.parent / relative
        parent = config_path.parent
        while parent != self.policy_path.parent:
            if parent.is_symlink():
                raise RemediationError("配置路径不能经过符号链接。")
            parent = parent.parent
        container = service.get("container")
        if not isinstance(container, str) or not CONTAINER_NAME.fullmatch(container):
            raise RemediationError("策略中的容器名无效。")
        health_url = service.get("health_url")
        parsed = urlparse(health_url) if isinstance(health_url, str) else None
        if (not parsed or parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.fragment):
            raise RemediationError("健康检查必须是本机 HTTP 地址。")
        allowed = service.get("allowed_keys")
        if not isinstance(allowed, dict) or not allowed:
            raise RemediationError("策略未列出可改配置字段。")
        verify_fields = service.get("verify_fields", {})
        if (not isinstance(verify_fields, dict) or
                any(key not in allowed or not isinstance(body_key, str) or not body_key
                    for key, body_key in verify_fields.items())):
            raise RemediationError("健康检查字段映射无效。")
        expected_status, max_latency = service.get("expected_status", 200), service.get("max_latency_ms", 2000)
        if (type(expected_status) is not int or not 100 <= expected_status <= 599
                or type(max_latency) not in {int, float} or not 0 < max_latency <= 30_000):
            raise RemediationError("健康检查阈值无效。")
        return service, config_path, _sha(policy_bytes)

    @staticmethod
    def _validate_changes(changes, allowed):
        if not isinstance(changes, dict) or not changes or len(changes) > 10:
            raise RemediationError("一次只能修改 1 到 10 个白名单字段。")
        for key, value in changes.items():
            if re.search(r"secret|token|password|credential|api.key", key, re.IGNORECASE):
                raise RemediationError("敏感凭据字段不能进入配置变更计划。")
            rule = allowed.get(key)
            if not isinstance(rule, dict):
                raise RemediationError(f"字段不在白名单中：{key}")
            kind = rule.get("type")
            if kind == "integer":
                valid = type(value) is int
            elif kind == "number":
                valid = type(value) in {int, float} and value == value and abs(value) != float("inf")
            elif kind == "string":
                valid = isinstance(value, str) and len(value) <= 200
            elif kind == "boolean":
                valid = type(value) is bool
            else:
                raise RemediationError(f"字段 {key} 的类型策略无效。")
            choices = rule.get("choices")
            if choices is not None and (not isinstance(choices, list) or not choices):
                raise RemediationError(f"字段 {key} 的候选值策略无效。")
            if not valid or choices is not None and value not in choices:
                raise RemediationError(f"字段 {key} 的值不符合白名单。")
            if kind in {"integer", "number"} and (value < rule.get("min", value)
                                                  or value > rule.get("max", value)):
                raise RemediationError(f"字段 {key} 超出允许范围。")

    def prepare(self, service_name, changes):
        service, config_path, policy_hash = self._service(service_name)
        self._validate_changes(changes, service["allowed_keys"])
        original = _read_regular(config_path)
        config = _json_object(original)
        if not set(changes).issubset(config):
            raise RemediationError("配置文件中缺少计划修改的字段。")
        if all(config[key] == value for key, value in changes.items()):
            raise RemediationError("配置已是目标值，无需变更。")
        container_id = self.adapter.container_id(service["container"])
        plan = {"id": uuid4().hex, "service": service_name, "config_file": str(config_path),
                "policy_hash": policy_hash, "config_sha256": _sha(original),
                "container": service["container"], "container_id": container_id,
                "health_url": service["health_url"], "expected_status": service.get("expected_status", 200),
                "max_latency_ms": service.get("max_latency_ms", 2000),
                "verify_fields": service.get("verify_fields", {}),
                "before": {key: config[key] for key in changes}, "changes": changes,
                "created_at": time.time()}
        with sqlite3.connect(self.db_path) as db:
            db.execute("INSERT INTO changes (id, plan, status) VALUES (?, ?, 'planned')",
                       (plan["id"], json.dumps(plan, ensure_ascii=False)))
        return plan

    def show(self, plan_id):
        with sqlite3.connect(self.db_path) as db:
            row = db.execute("SELECT plan, status, result FROM changes WHERE id=?", (plan_id,)).fetchone()
        if not row:
            raise RemediationError("变更计划不存在。")
        return {"plan": json.loads(row[0]), "status": row[1], "result": json.loads(row[2]) if row[2] else None}

    def approve(self, plan_id, confirmation):
        with sqlite3.connect(self.db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT plan, status FROM changes WHERE id=?", (plan_id,)).fetchone()
            if not row or row[1] != "planned":
                raise RemediationError("计划不存在或已审批。")
            plan = json.loads(row[0])
            if confirmation != plan["service"]:
                raise RemediationError("服务名确认不一致，未签发令牌。")
            service, config_path, policy_hash = self._service(plan["service"])
            if (policy_hash != plan["policy_hash"] or str(config_path) != plan["config_file"]
                    or service["container"] != plan["container"]):
                raise RemediationError("变更策略已变化，请重新生成计划。")
            token = secrets.token_urlsafe(32)
            db.execute("UPDATE changes SET status='approved', token_hash=?, token_expires=? WHERE id=?",
                       (_sha(token.encode()), time.time() + TOKEN_TTL_SECONDS, plan_id))
        return token

    def _finish(self, plan_id, status, result):
        with sqlite3.connect(self.db_path) as db:
            db.execute("UPDATE changes SET status=?, result=? WHERE id=?",
                       (status, json.dumps(result, ensure_ascii=False), plan_id))
        return result

    @contextmanager
    def _apply_lock(self):
        fd = os.open(self.state_dir / "apply.lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def apply(self, plan_id, token):
        with self._apply_lock():
            return self._apply_locked(plan_id, token)

    def _apply_locked(self, plan_id, token):
        with sqlite3.connect(self.db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT plan, status, token_hash, token_expires FROM changes WHERE id=?", (plan_id,)).fetchone()
            if (not row or row[1] != "approved" or not isinstance(token, str)
                    or not secrets.compare_digest(row[2] or "", _sha(token.encode()))
                    or time.time() > row[3]):
                raise RemediationError("一次性审批令牌无效、已使用或已过期。")
            plan = json.loads(row[0])
            db.execute("UPDATE changes SET status='executing', token_hash=NULL, token_expires=NULL WHERE id=?",
                       (plan_id,))
        backup_path = None
        stage = "preflight"
        try:
            service, config_path, policy_hash = self._service(plan["service"])
            if (policy_hash != plan["policy_hash"] or str(config_path) != plan["config_file"]
                    or service["container"] != plan["container"]):
                raise RemediationError("策略已变化，计划失效。")
            original = _read_regular(config_path)
            if _sha(original) != plan["config_sha256"]:
                raise RemediationError("配置文件已变化，计划失效。")
            if self.adapter.container_id(plan["container"]) != plan["container_id"]:
                raise RemediationError("目标容器已变化，计划失效。")
            updated = _json_object(original)
            self._validate_changes(plan["changes"], service["allowed_keys"])
            updated.update(plan["changes"])
            backup_dir = self.state_dir / "backups"
            backup_dir.mkdir(mode=0o700, exist_ok=True)
            if backup_dir.is_symlink():
                raise RemediationError("备份目录不能是符号链接。")
            backup_path = backup_dir / f"{plan_id}.json"
            fd = os.open(backup_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(original)
                stream.flush()
                os.fsync(stream.fileno())
            stage = "write_config"
            _atomic_replace(config_path, (json.dumps(updated, ensure_ascii=False, indent=2) + "\n").encode(),
                            stat.S_IMODE(config_path.stat().st_mode), plan["config_sha256"])
            stage = "restart"
            self.adapter.restart(plan["container"], plan["container_id"])
            stage = "verify"
            observations = []
            for attempt in range(3):
                try:
                    observed = self.adapter.health(plan["health_url"])
                    expected_fields = plan["verify_fields"]
                    fields_ok = all(observed["body"].get(body_key) == plan["changes"][config_key]
                                    for config_key, body_key in expected_fields.items()
                                    if config_key in plan["changes"])
                    healthy = (observed["status_code"] == plan["expected_status"]
                               and observed["latency_ms"] <= plan["max_latency_ms"] and fields_ok)
                    observations.append({"attempt": attempt + 1, **observed, "healthy": healthy})
                    if healthy:
                        return self._finish(plan_id, "verified", {"status": "verified", "stage": stage,
                                            "backup_path": str(backup_path), "observations": observations})
                except RemediationError as exc:
                    observations.append({"attempt": attempt + 1, "error": str(exc), "healthy": False})
                if attempt < 2:
                    time.sleep(1)
            return self._finish(plan_id, "verification_failed", {"status": "verification_failed",
                                "stage": stage, "backup_path": str(backup_path), "observations": observations})
        except (OSError, RemediationError, ValueError) as exc:
            return self._finish(plan_id, "failed", {"status": "failed", "stage": stage,
                                "error": str(exc), "backup_path": str(backup_path) if backup_path else None})
