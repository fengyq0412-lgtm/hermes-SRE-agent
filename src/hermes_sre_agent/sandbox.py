"""一次性 Docker 测试沙箱：只接受项目相对测试路径，不接受 Shell。"""

import ast
import os
import selectors
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4


IMAGE = "hermes-sandbox:local"
OUTPUT_LIMIT = 32_000


class DockerSandbox:
    def __init__(self, root, timeout=60):
        self.root = Path(root).resolve()
        self.timeout = timeout

    def discover_tests(self, preferred="tests"):
        """静态识别 pytest 测试入口，不导入项目；缺少默认目录时查找其他测试位置。"""
        from .code_review import SourceTools
        tests = []
        for name in SourceTools(self.root).list_files()["files"]:
            path = Path(name)
            if path.suffix != ".py" or not (path.name.startswith("test_") or path.name.endswith("_test.py")):
                continue
            try:
                tree = ast.parse(SourceTools(self.root).resolve(name).read_text(encoding="utf-8"))
            except SyntaxError:
                tests.append(name)  # 测试语法错误也必须交给验证报告，不能静默过滤。
                continue
            if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
                   or isinstance(node, ast.ClassDef) and any(isinstance(child, ast.FunctionDef) and child.name.startswith("test_")
                                                            for child in node.body) for node in tree.body):
                tests.append(name)
        if preferred != "tests":
            return [preferred]
        return tests[:50]

    def snapshot(self, destination):
        from .code_review import SourceTools

        tools = SourceTools(self.root)
        catalog = tools.list_files()
        if catalog["truncated"]:
            raise ValueError("项目超过沙箱快照上限，请选择更小的项目目录。")
        size = 0
        # 使用目录描述符逐层打开，避免复制过程中符号链接替换导致越界读取。
        for name in catalog["files"]:
            fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                parts = Path(name).parts
                for part in parts[:-1]:
                    next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    os.close(fd)
                    fd = next_fd
                source = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                with os.fdopen(source, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise ValueError("沙箱不复制特殊文件或硬链接。")
                    data = stream.read(256_001)
                size += len(data)
                if len(data) > 256_000 or size > 20_000_000:
                    raise ValueError("源码快照超过大小限制（单文件 256 KB，总计 20 MB）。")
                target = Path(destination) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(0o644)
            finally:
                os.close(fd)
        return len(catalog["files"])

    def run_tests(self, path="tests"):
        paths = path if isinstance(path, list) else [path]
        if not 1 <= len(paths) <= 50 or any(not isinstance(p, str) or not p or p.startswith("-")
                or Path(p).is_absolute() or ".." in Path(p).parts for p in paths):
            raise ValueError("测试路径必须是项目内的相对路径。")
        name = "hermes-test-" + uuid4().hex
        with tempfile.TemporaryDirectory(prefix="hermes-sandbox-") as directory:
            count = self.snapshot(directory)
            if any(not (Path(directory) / p).exists() for p in paths):
                raise ValueError("测试路径不存在或已被快照过滤，请先定位测试文件。")
            Path(directory).chmod(0o755)
            command = ["docker", "create", "--pull=never", "--name", name,
                       "--network", "none", "--read-only", "--cap-drop", "ALL",
                       "--security-opt", "no-new-privileges", "--user", "65534:65534",
                       "--memory", "512m", "--memory-swap", "512m", "--cpus", "1",
                       "--pids-limit", "64", "--log-driver", "none",
                       "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=128m,mode=1777",
                       "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
                       "--env", "HOME=/tmp", "--mount", f"type=bind,src={directory},dst=/source,readonly",
                       IMAGE, *paths]
            created_ok = False
            try:
                created = subprocess.run(command, capture_output=True, timeout=15)
                if created.returncode:
                    raise ValueError("无法创建沙箱：请启动 Docker Desktop，并先构建 hermes-sandbox:local 镜像。")
                created_ok = True
                return self._execute(name, count)
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                raise ValueError("Docker 未安装或响应超时，沙箱未能完成。") from exc
            finally:
                # 只清理本次随机命名的容器；超时后也不会留下后台执行的项目代码。
                try:
                    removed = subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
                    if removed.returncode and created_ok:
                        raise ValueError(f"沙箱清理失败，请检查容器 {name}。")
                except (OSError, subprocess.TimeoutExpired):
                    if created_ok:
                        raise ValueError(f"沙箱清理失败，请检查容器 {name}。")

    def _execute(self, name, count):
        process = subprocess.Popen(["docker", "start", "-a", name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = bytearray()
        status = "finished"
        deadline = time.monotonic() + self.timeout
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        status = "timeout"
                        break
                    if not selector.select(min(.2, remaining)):
                        continue
                    chunk = os.read(process.stdout.fileno(), 4096)
                    if not chunk:
                        break
                    output.extend(chunk[:OUTPUT_LIMIT - len(output)])
                    if len(output) >= OUTPUT_LIMIT:
                        status = "output_limit"
                        break
            if status == "finished":
                process.wait(timeout=max(.1, deadline - time.monotonic()))
            return {"status": status, "exit_code": process.returncode if status == "finished" else None,
                    "output": output.decode("utf-8", errors="replace"), "snapshot_files": count,
                    "scope": "过滤后的临时副本；无网络；未安装项目额外依赖"}
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
