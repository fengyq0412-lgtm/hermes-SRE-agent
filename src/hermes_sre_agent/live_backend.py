"""本机只读采集后端。

所有子进程命令都在此处固定定义；外部输入只能进入经过校验的参数位，
从设计上避免把用户文本拼接为 Shell 命令。
"""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .tools import ToolExecutionError


class _NoHealthRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


_HEALTH_OPENER = build_opener(_NoHealthRedirect())


class LiveReadOnlyBackend:
    """直接读取一台 Linux/macOS 主机；缺少 Docker/GPU 时会明确报告失败。"""

    _container_name = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

    def __init__(self, repository: Optional[Path] = None, timeout_seconds: int = 5):
        self.repository = (repository or Path.cwd()).resolve()
        self.timeout_seconds = timeout_seconds
        self._handlers: Dict[str, Callable[..., Any]] = {
            "system_metrics": self._system_metrics,
            "gpu_metrics": self._gpu_metrics,
            "docker_list": self._docker_list,
            "docker_logs": self._docker_logs,
            "service_health": self._service_health,
            "git_recent_changes": self._git_recent_changes,
            "config_diff": self._config_diff,
            "git_code_review": self._git_code_review,
        }

    def read(self, tool: str, **arguments: Any) -> Any:
        handler = self._handlers.get(tool)
        if handler is None:
            raise ToolExecutionError(f"本机后端不支持只读工具：{tool}")
        return handler(**arguments)

    def _run(self, command: List[str]) -> str:
        """运行固定命令列表，不通过 shell 解析命令字符串。"""
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise ToolExecutionError(f"未安装命令：{command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ToolExecutionError(f"采集超时（{self.timeout_seconds} 秒）：{command[0]}") from exc
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "未知错误"
            raise ToolExecutionError(f"{' '.join(command[:2])} 执行失败：{message[:240]}")
        # Docker 把容器标准错误日志写到命令的 stderr；成功时两路都是应用证据。
        return completed.stdout + completed.stderr if command[:2] == ["docker", "logs"] else completed.stdout

    def _system_metrics(self) -> Dict[str, Any]:
        disk = shutil.disk_usage(self.repository)
        load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
        cpu_count = os.cpu_count() or 1
        linux_cpu = self._linux_cpu_used_percent()
        memory_percent = self._memory_used_percent()
        return {
            "cpu_percent": linux_cpu if linux_cpu is not None else "未采集",
            "cpu_measurement": "Linux /proc/stat 100ms 采样" if linux_cpu is not None else "当前系统未提供 CPU 使用率采样",
            "load_based_estimate_percent": min(round(load[0] / cpu_count * 100, 1), 100.0),
            "memory_used_percent": memory_percent,
            "disk_used_percent": round(disk.used / disk.total * 100, 1),
            "load_average": [round(value, 2) for value in load],
        }

    @staticmethod
    def _linux_cpu_used_percent() -> Optional[float]:
        def snapshot() -> tuple[int, int]:
            parts = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()
            if parts[0] != "cpu" or len(parts) < 5:
                raise ValueError("/proc/stat 缺少 CPU 数据")
            values = [int(part) for part in parts[1:9]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            return sum(values), idle

        try:
            first_total, first_idle = snapshot()
            time.sleep(0.1)
            second_total, second_idle = snapshot()
        except (OSError, ValueError, IndexError):
            return None
        delta = second_total - first_total
        return round((1 - (second_idle - first_idle) / delta) * 100, 1) if delta > 0 else None

    def _memory_used_percent(self) -> Any:
        meminfo = Path("/proc/meminfo")
        if not meminfo.exists():
            return "不可用（当前系统未提供 /proc/meminfo）"
        values = {}
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0])
        except (OSError, ValueError, IndexError):
            return "不可用（无法读取 /proc/meminfo）"
        total, available = values.get("MemTotal", 0), values.get("MemAvailable", 0)
        return round((total - available) / total * 100, 1) if total else "不可用"

    def _gpu_metrics(self) -> Dict[str, Any]:
        output = self._run([
            "nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]).strip()
        if not output:
            raise ToolExecutionError("nvidia-smi 未返回 GPU 数据")
        name, utilization, used, total = [part.strip() for part in output.splitlines()[0].split(",")]
        return {"gpu": name, "gpu_utilization": int(utilization), "memory_used_mb": int(used), "memory_total_mb": int(total)}

    def _docker_list(self) -> List[Dict[str, Any]]:
        output = self._run(["docker", "ps", "-a", "--format", "{{json .}}"])
        containers = []
        for line in output.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            state = item.get("State", "").lower()
            status = item.get("Status", "")
            containers.append({"name": item.get("Names", ""),
                               "status": state or ("running" if status.startswith("Up") else status.lower() or "unknown"),
                               "detail": status, "cpu": "未采集", "memory": "未采集"})
        if not any(item["status"] == "running" for item in containers):
            return containers
        try:
            stats = self._run(["docker", "stats", "--no-stream", "--format", "{{json .}}"])
            by_name = {item["name"]: item for item in containers}
            for line in stats.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                target = by_name.get(item.get("Name"))
                if target and target["status"] == "running":
                    target["cpu"] = item.get("CPUPerc", "未采集")
                    target["memory"] = item.get("MemUsage", "未采集")
                    target["memory_percent"] = item.get("MemPerc", "未采集")
        except (ToolExecutionError, ValueError) as exc:
            for item in containers:
                if item["status"] == "running":
                    item["resource_error"] = str(exc)
        return containers

    def _docker_logs(self, container: str, tail: int = 200, **_: Any) -> Dict[str, List[str]]:
        if not self._container_name.fullmatch(container):
            raise ToolExecutionError("容器名仅允许字母、数字、点、下划线和连字符")
        try:
            safe_tail = min(max(int(tail), 1), 200)
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError("日志行数必须为 1 到 200 的整数") from exc
        output = self._run(["docker", "logs", "--tail", str(safe_tail), "--", container])
        return {"lines": output.splitlines()}

    def _service_health(self, url: str, **_: Any) -> Dict[str, Any]:
        parsed = urlparse(url) if isinstance(url, str) else None
        if (not parsed or parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.fragment):
            raise ToolExecutionError("健康检查地址仅支持 http 或 https")
        started = time.perf_counter()
        try:
            with _HEALTH_OPENER.open(Request(url, method="GET"), timeout=self.timeout_seconds) as response:
                status = response.status
        except HTTPError as exc:
            # 4xx/5xx 是服务状态的重要证据，并不等同于采集工具失败。
            return {"status_code": exc.code, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
        except (URLError, ValueError, TimeoutError) as exc:
            raise ToolExecutionError(f"健康检查失败：{exc}") from exc
        return {"status_code": status, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

    def _git_recent_changes(self) -> Dict[str, Any]:
        raw = self._run(["git", "-C", str(self.repository), "log", "-5", "--format=%h%x00%s%x00%ci", "--name-only"])
        commits = []
        for line in raw.splitlines():
            if "\x00" in line:
                header = line.split("\x00")
                if len(header) != 3:
                    raise ToolExecutionError("无法解析最近一次 Git 提交")
                commits.append({"sha": header[0], "message": header[1], "date": header[2], "files": []})
            elif line and commits and len(commits[-1]["files"]) < 100:
                commits[-1]["files"].append(line)
        if not commits:
            raise ToolExecutionError("无法解析最近一次 Git 提交")
        return {
            "summary": f"最近提交 {commits[0]['sha']}：{commits[0]['message']}（{commits[0]['date']}）；共读取 {len(commits)} 次提交",
            "commits": commits,
        }

    def _latest_git_diff(self) -> str:
        # git show 兼容只有初次提交的仓库；不运行外部 diff 或分页器。
        return self._run([
            "git", "--no-pager", "-c", "core.pager=cat", "-C", str(self.repository),
            "show", "--format=", "--no-ext-diff", "--no-renames", "HEAD", "--",
        ])

    def _config_diff(self) -> Dict[str, Any]:
        diff = self._latest_git_diff()
        changes = []
        current_path = None
        removed = []
        added = []
        pattern = re.compile(r"^[+-]\s*['\"]?batch_size['\"]?\s*[:=]\s*(\d+)\b")

        def flush() -> None:
            if current_path and len(removed) == len(added) == 1:
                changes.append({"path": current_path, "key": "batch_size", "before": removed[0], "after": added[0]})

        for line in diff.splitlines():
            if line.startswith("diff --git "):
                flush()
                match = re.match(r"diff --git a/(.+) b/(.+)$", line)
                path = match.group(2) if match else ""
                current_path = path if Path(path).suffix.lower() in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"} else None
                removed, added = [], []
            elif line.startswith("@@"):
                flush()
                removed, added = [], []
            elif current_path and not line.startswith(("---", "+++")):
                match = pattern.match(line)
                if match:
                    if line.startswith("-"):
                        removed.append(int(match.group(1)))
                    else:
                        added.append(int(match.group(1)))
        flush()
        return {"path": changes[0]["path"] if changes else "未发现已提交的配置字段变更",
                "changes": changes, "raw_diff_available": bool(diff.strip()),
                "source": "repository_latest_commit", "runtime_config_verified": False}

    def _git_code_review(self) -> Dict[str, Any]:
        """仅返回最新提交差异的有限片段，既不执行代码也不暴露整个仓库。"""
        diff = self._latest_git_diff()
        limit = 20_000
        return {
            "base": "HEAD 的父提交（初次提交时为空树）",
            "head": "HEAD",
            "diff": diff[:limit],
            "truncated": len(diff) > limit,
        }
