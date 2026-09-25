"""修复编排与人工审批：模型仅编辑快照，批准后才写回原项目。"""

import difflib
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .code_review import CodeReviewAgent, SourceTools
from .sandbox import DockerSandbox
from .model_client import ModelRequestError


def open_parent(root, name, created_dirs=None):
    """逐层拒绝符号链接，返回调用方负责关闭的父目录描述符。"""
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not SourceTools.allowed(relative):
        raise ValueError("补丁路径不属于允许的项目源码。")
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in relative.parts[:-1]:
            if created_dirs is not None:
                try:
                    os.mkdir(part, mode=0o755, dir_fd=fd)
                    # 用目录自身的描述符做失败后的空目录清理，避免跟随替换的链接。
                    created_dirs.append((os.dup(fd), part))
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd, relative.name
    except BaseException:
        os.close(fd)
        raise


def read_at(fd, name):
    source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    with os.fdopen(source, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("不修改特殊文件或硬链接。")
        return stream.read(256_001), stat.S_IMODE(info.st_mode)


def replace_at(fd, name, content, mode):
    temporary = ".hermes-" + uuid4().hex
    target = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode, dir_fd=fd)
    try:
        with os.fdopen(target, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), mode)
        os.replace(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=fd)
        except FileNotFoundError:
            pass


def create_at(fd, name, content, mode=0o644):
    """先写完整临时文件，再以不覆盖目标的方式发布；冲突时保留现有文件。"""
    temporary = ".hermes-" + uuid4().hex
    target = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode, dir_fd=fd)
    try:
        with os.fdopen(target, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
    finally:
        os.unlink(temporary, dir_fd=fd)


def is_test_file(name):
    path = Path(name)
    return path.suffix == ".py" and (path.name.startswith("test_") or path.name.endswith("_test.py"))


def editable_path(path):
    if not isinstance(path, str):
        raise ValueError("文件路径必须是字符串。")
    relative = Path(path)
    if (relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py"
            or not SourceTools.allowed(relative) or relative.as_posix() != path):
        raise ValueError("仅允许项目内规范相对路径的 Python 文件，不能访问隐藏文件、密钥或越界路径。")
    return relative


class ChangeSet:
    def __init__(self, workspace):
        self.workspace = Path(workspace)
        self.originals = {}
        self.read_ranges = {}
        self.test_only = False

    def _check_path(self, path):
        relative = editable_path(path)
        if self.test_only and not (is_test_file(path) or "tests" in relative.parts or relative.name == "conftest.py"):
            raise ValueError("测试补充阶段只能创建或修改测试文件。")
        return relative

    def create_file(self, path, content):
        self._check_path(path)
        if not isinstance(content, str) or not content.strip() or len(content.encode()) > 32_000:
            raise ValueError("新文件需包含非空源码，最多 32 KB。")
        if len(self.originals) >= 5 and path not in self.originals:
            raise ValueError("一次最多修改或创建 5 个文件。")
        created_dirs = []
        try:
            fd, name = open_parent(self.workspace, path, created_dirs)
            try:
                create_at(fd, name, content.encode())
            finally:
                os.close(fd)
            self.originals[path] = None
        finally:
            for parent_fd, _ in created_dirs:
                os.close(parent_fd)
        return {"path": path, "operation": "create", "message": "新文件仅创建在临时副本中，待人工批准后写回。"}

    def allow_range(self, path, start_line, end_line):
        """记录实际已返回给模型的行号，不接受模型自报的阅读范围。"""
        digest = hashlib.sha256(SourceTools(self.workspace).resolve(path).read_bytes()).hexdigest()
        self.read_ranges.setdefault(path, []).append((start_line, end_line, digest))

    def edit_file(self, path, old_text, new_text):
        relative = self._check_path(path)
        path = relative.as_posix()
        if not all(isinstance(value, str) for value in (old_text, new_text)) or not old_text:
            raise ValueError("请提供非空 old_text 和字符串 new_text。")
        if len(new_text.encode()) > 32_000 or len(self.originals) >= 5 and path not in self.originals:
            raise ValueError("单段修改最多 32 KB，一次最多修改 5 个文件。")
        target = SourceTools(self.workspace).resolve(path)
        before = target.read_bytes()
        if target.stat().st_nlink != 1:
            raise ValueError("不修改硬链接。")
        text = before.decode("utf-8")
        if text.count(old_text) != 1:
            raise ValueError("old_text 必须在当前文件中精确出现一次，请重新读取相关源码。")
        updated = text.replace(old_text, new_text, 1).encode("utf-8")
        if not updated.strip():
            raise ValueError("不允许通过清空文件变相删除源码。")
        if len(updated) > 256_000:
            raise ValueError("修改后的文件超过 256 KB。")
        self.originals.setdefault(path, before)
        target.write_bytes(updated)
        return {"path": path, "message": "修改已写入临时副本，尚未应用到原项目。"}

    def edit_lines(self, path, start_line, end_line, new_text):
        """只在已读取范围允许的行上操作；行范围校验由 Agent 完成。"""
        relative = self._check_path(path)
        if (type(start_line) is not int or type(end_line) is not int or
                start_line < 1 or end_line < start_line or end_line - start_line >= 200):
            raise ValueError("行号替换必须指定 1 到 200 行的有效范围。")
        if not isinstance(new_text, str) or len(new_text.encode()) > 32_000:
            raise ValueError("替换内容必须是最多 32 KB 的字符串。")
        path = relative.as_posix()
        if len(self.originals) >= 5 and path not in self.originals:
            raise ValueError("一次最多修改 5 个文件。")
        target = SourceTools(self.workspace).resolve(path)
        before = target.read_bytes()
        digest = hashlib.sha256(before).hexdigest()
        if not any(start <= start_line and end_line <= end and version == digest
                   for start, end, version in self.read_ranges.get(path, [])):
            raise ValueError("行号替换必须落在当前版本的已读范围内，修改后需重新读取。")
        if target.stat().st_nlink != 1:
            raise ValueError("不修改硬链接。")
        lines = before.decode("utf-8").splitlines(keepends=True)
        if end_line > len(lines):
            raise ValueError("行号超出当前文件范围，请重新读取源码。")
        if end_line < len(lines) and new_text and not new_text.endswith(("\n", "\r")):
            raise ValueError("替换非文件末尾行时，new_text 必须以换行符结束。")
        updated = ("".join(lines[:start_line - 1]) + new_text + "".join(lines[end_line:])).encode("utf-8")
        if not updated.strip():
            raise ValueError("不允许通过清空文件变相删除源码。")
        if len(updated) > 256_000:
            raise ValueError("修改后的文件超过 256 KB。")
        self.originals.setdefault(path, before)
        target.write_bytes(updated)
        return {"path": path, "message": "行号补丁已写入临时副本，尚未应用到原项目。"}

    def changes(self):
        return {name: (before, (self.workspace / name).read_bytes()) for name, before in self.originals.items()
                if before != (self.workspace / name).read_bytes()}

    def apply_edits(self, edits):
        """批量补丁只作用于临时副本；任一替换失败时恢复整批修改。"""
        if not isinstance(edits, list) or not 1 <= len(edits) <= 12:
            raise ValueError("一次补丁需要 1 到 12 个精确替换。")
        originals = dict(self.originals)
        snapshots = {}
        line_paths = set()
        for edit in edits:
            if not isinstance(edit, dict) or set(edit) not in (
                    {"path", "old_text", "new_text"}, {"path", "start_line", "end_line", "new_text"}, {"path", "content"}):
                raise ValueError("每项修改必须使用精确文本替换或已读行号范围替换。")
            if "start_line" in edit:
                if edit["path"] in line_paths or sum(isinstance(item, dict) and item.get("path") == edit["path"] for item in edits) > 1:
                    raise ValueError("同一文件的一次批量补丁最多包含一个行号替换。")
                line_paths.add(edit["path"])
            self._check_path(edit["path"])
            if "content" in edit:
                if sum(isinstance(item, dict) and item.get("path") == edit["path"] for item in edits) > 1:
                    raise ValueError("新文件在同一批补丁中只能提交一次完整内容。")
                target = self.workspace / edit["path"]
                snapshots.setdefault(target, None)
            else:
                target = SourceTools(self.workspace).resolve(edit["path"])
                snapshots.setdefault(target, target.read_bytes())
        try:
            return [self.create_file(**edit) if "content" in edit else
                    self.edit_lines(**edit) if "start_line" in edit else self.edit_file(**edit) for edit in edits]
        except Exception:
            for target, content in snapshots.items():
                if content is None:
                    # 仅删除本批已成功创建的临时文件；创建失败的已有路径绝不能删除。
                    path = target.relative_to(self.workspace).as_posix()
                    if path not in originals and path in self.originals and self.originals[path] is None:
                        target.unlink()
                else:
                    target.write_bytes(content)
            self.originals = originals
            raise


class PatchProposal:
    def __init__(self, changes, validation, baseline=None):
        for name, (before, after) in changes.items():
            editable_path(name)
            if before is not None and not isinstance(before, bytes) or not isinstance(after, bytes) or not after.strip():
                raise ValueError("方案只能包含创建或修改，不能删除或清空文件。")
        self.id = uuid4().hex
        self.changes = changes
        self.validation = validation
        self.baseline = baseline
        self.state = "pending"

    def public(self):
        files = []
        for name, (before, after) in self.changes.items():
            diff = "".join(difflib.unified_diff((before or b"").decode().splitlines(True), after.decode().splitlines(True),
                                               fromfile="/dev/null" if before is None else "a/" + name, tofile="b/" + name))
            files.append({"path": name, "diff": diff, "operation": "create" if before is None else "modify",
                          "is_test": is_test_file(name) or "tests" in Path(name).parts or Path(name).name == "conftest.py",
                          "base_sha256": hashlib.sha256(before).hexdigest() if before is not None else None})
        return {"id": self.id, "files": files, "validation": self.validation,
                "can_apply": self.state == "pending" and self.validation.get("status") == "finished" and self.validation.get("exit_code") == 0,
                "state": self.state}

    def apply(self, root, backup_root):
        if self.state != "pending" or not self.public()["can_apply"]:
            raise ValueError("方案已处理或沙箱测试未通过，不能应用。")
        handles, applied, created_dirs = {}, [], []
        backup = Path(backup_root) / self.id
        manifest = None
        try:
            if self.baseline is not None:
                catalog = SourceTools(root).list_files()
                if catalog["truncated"] or set(catalog["files"]) != set(self.baseline):
                    raise ValueError("项目文件列表已变化，请重新生成并验证方案。")
                for name, expected in self.baseline.items():
                    fd, basename = open_parent(root, name)
                    try:
                        if hashlib.sha256(read_at(fd, basename)[0]).hexdigest() != expected:
                            raise ValueError(f"{name} 已变化，原沙箱验证结果已过期。")
                    finally:
                        os.close(fd)
            # 写入前整体检查，避免覆盖生成方案之后用户新改动的文件。
            for name, (before, after) in self.changes.items():
                fd, basename = open_parent(root, name, created_dirs if before is None else None)
                handles[name] = (fd, basename)
                if before is None:
                    try:
                        os.stat(basename, dir_fd=fd, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        raise ValueError(f"{name} 已存在，不能覆盖新建目标。")
                    mode = 0o644
                else:
                    current, mode = read_at(fd, basename)
                    if current != before:
                        raise ValueError(f"{name} 已变化，需重新生成方案。")
                handles[name] = (fd, basename, mode)
            backup.mkdir(parents=True, exist_ok=False)
            for name, (before, _) in self.changes.items():
                if before is None:
                    continue
                target = backup / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(before)
            manifest = {"proposal_id": self.id, "project": str(root), "files": list(self.changes),
                        "decision": "approved", "state": "applying",
                        "approved_at": datetime.now(timezone.utc).isoformat(),
                        "operations": {name: "create" if before is None else "modify" for name, (before, _) in self.changes.items()},
                        "sha256": {name: {"before": hashlib.sha256(before).hexdigest() if before is not None else None,
                                          "after": hashlib.sha256(after).hexdigest()}
                                   for name, (before, after) in self.changes.items()}}
            (backup / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            for name, (before, after) in self.changes.items():
                fd, basename, mode = handles[name]
                if before is None:
                    create_at(fd, basename, after, mode)
                else:
                    if read_at(fd, basename)[0] != before:
                        raise ValueError(f"{name} 在应用前发生变化，请重新生成。")
                    replace_at(fd, basename, after, mode)
                applied.append(name)
            manifest["state"] = "applied"
            (backup / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            self.state = "applied"
            return str(backup)
        except Exception as exc:
            self.state = "invalid"
            # 仅回滚仍保持本次写入内容的文件，避免覆盖外部并发修改。
            unresolved = []
            for name in reversed(applied):
                fd, basename, mode = handles[name]
                before, after = self.changes[name]
                try:
                    if read_at(fd, basename)[0] == after:
                        if before is None:
                            os.unlink(basename, dir_fd=fd)
                        else:
                            replace_at(fd, basename, before, mode)
                    else:
                        unresolved.append(name)
                except (OSError, ValueError):
                    unresolved.append(name)
            if backup.exists() and manifest is not None:
                try:
                    manifest["state"] = "manual_recovery_required" if unresolved else "rolled_back"
                    (backup / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
                except OSError:
                    pass
            if unresolved:
                self.state = "failed"
                raise ValueError(f"应用失败，需人工恢复 {unresolved}；原文件备份：{backup}") from exc
            raise
        finally:
            for handle in handles.values():
                os.close(handle[0])
            for fd, name in reversed(created_dirs):
                if self.state != "applied":
                    try:
                        os.rmdir(name, dir_fd=fd)
                    except OSError:
                        pass  # 已被他人写入内容的目录保留，不能递归删除。
                os.close(fd)


class RepairWorkflow:
    def __init__(self, root, client, on_event, on_trace=None, history_search=None):
        self.root, self.client, self.on_event = root, client, on_event
        self.on_trace = on_trace or (lambda entry: None)
        self.history_search = history_search

    def run(self, question, history, test_path="tests"):
        self.on_event({"message": "修复编排 · 创建隔离源码快照", "kind": "stage"})
        self.on_trace({"stage": "修复", "event": "snapshot_start", "summary": "创建隔离源码快照", "details": {}})
        with tempfile.TemporaryDirectory(prefix="hermes-repair-") as workspace:
            DockerSandbox(self.root).snapshot(workspace)
            baseline = {name: hashlib.sha256((Path(workspace) / name).read_bytes()).hexdigest()
                        for name in SourceTools(workspace).list_files()["files"]}
            self.on_trace({"stage": "修复", "event": "snapshot_ready", "summary": "隔离快照已建立",
                           "details": {"file_count": len(baseline)}})
            changes = ChangeSet(workspace)
            agent = CodeReviewAgent(workspace, self.client, self.on_event, editor=changes, on_trace=self.on_trace,
                                    history_search=self.history_search)
            result = agent.run(question, history)
            patch = changes.changes()
            if not patch:
                self.on_trace({"stage": "修复", "event": "no_patch", "summary": "未生成可审核的补丁，原项目未修改", "details": {}})
                if result.get("outcome") not in {"blocked", "needs_input"}:
                    result = agent._blocked_repair(result["steps"], "执行结束但没有检测到文件净变更。")
                return result, None
            paths = DockerSandbox(workspace).discover_tests(test_path)
            if not paths:
                self.on_event({"message": "未发现可运行测试，正在为本次修改补充回归测试", "kind": "stage"})
                self.on_trace({"stage": "测试", "event": "test_generation_start", "summary": "自动补充回归测试",
                               "details": {"changed_files": sorted(patch)}})
                changes.test_only = True
                try:
                    test_agent = CodeReviewAgent(workspace, self.client, self.on_event, editor=changes,
                                                 on_trace=self.on_trace, max_tool_calls=6)
                    test_result = test_agent.run(
                        "为本轮修改创建针对性的 pytest 回归测试，使用 tests/test_*.py。先读取修改后的源码，"
                        "检查正常和错误分支；不得使用恒真断言、删断言、跳过测试来制造通过。"
                        "执行环境是 Python 3.11、pytest、无网络；外部数据库和网络调用应按业务边界 mock。"
                        "只能创建或修改测试文件，不得改业务源码。无法验证就说明依赖或规则阻碍。"
                        "当前用户目标：" + question + "\n变更文件：" + ", ".join(sorted(patch)),
                        history=[{"role": "user", "content": "本次候选修改摘要：" + result["answer"][:3000]}])
                    result["steps"].extend(test_result["steps"])
                except ModelRequestError:
                    self.on_trace({"stage": "测试", "event": "test_generation_failed", "summary": "测试生成调用未完成",
                                   "details": {}})
                finally:
                    changes.test_only = False
                paths = DockerSandbox(workspace).discover_tests(test_path)
                patch = changes.changes()
            result["outcome"] = "patch_ready"
            self.on_trace({"stage": "修复", "event": "patch_ready", "summary": "临时副本已生成补丁",
                           "details": {"file_count": len(patch), "files": sorted(patch)}})
            self.on_event({"message": "修复编排 · 在沙箱验证最终补丁", "kind": "stage"})
            self.on_trace({"stage": "验证", "event": "test_start", "summary": "开始在沙箱验证补丁",
                           "details": {"test_paths": paths}})
            try:
                if not paths:
                    raise ValueError("尚未生成可收集的回归测试，无法验证此补丁；原项目未修改。")
                validation = dict(DockerSandbox(workspace).run_tests(paths))
            except (ValueError, OSError) as exc:
                validation = {"status": "unavailable", "exit_code": None, "output": str(exc)}
            validation["test_paths"] = paths
            result["steps"].append({"tool": "run_tests", "arguments": {"path": paths}, "result": validation})
            self.on_trace({"stage": "验证", "event": "test_result", "summary": "沙箱验证结束",
                           "details": {"status": validation.get("status"), "exit_code": validation.get("exit_code")}})
            self.on_event({"message": "修复编排 · 等待人工审核 diff 与测试结果", "kind": "stage"})
            self.on_trace({"stage": "审批", "event": "approval_requested", "summary": "等待人工审核补丁与测试结果",
                           "details": {"file_count": len(patch), "can_apply": validation.get("status") == "finished" and validation.get("exit_code") == 0}})
            return result, PatchProposal(patch, validation, baseline)
