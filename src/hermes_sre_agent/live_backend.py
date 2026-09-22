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
from urllib.request import Request, urlopen

from .tools import ToolExecutionError


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
        return completed.stdout

    def _system_metrics(self) -> Dict[str, Any]:
        disk = shutil.disk_usage(self.repository)
        load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
        cpu_count = os.cpu_count() or 1
        # 标准库没有跨平台的瞬时 CPU 百分比；用一分钟负载给出明确标记的近似值。
        cpu_percent = min(round(load[0] / cpu_count * 100, 1), 100.0)
        memory_percent = self._memory_used_percent()
        return {
            "cpu_percent": cpu_percent,
            "cpu_measurement": "按一分钟负载估算",
            "memory_used_percent": memory_percent,
            "disk_used_percent": round(disk.used / disk.total * 100, 1),
            "load_average": [round(value, 2) for value in load],
        }

    def _memory_used_percent(self) -> Any:
        meminfo = Path("/proc/meminfo")
        if not meminfo.exists():
            return "不可用（当前系统未提供 /proc/meminfo）"
        values = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
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
        output = self._run(["docker", "ps", "--format", "{{json .}}"])
        containers = []
        for line in output.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            containers.append({
                "name": item.get("Names", ""),
                "status": "running" if item.get("Status", "").startswith("Up") else item.get("Status", "unknown"),
                "cpu": "未采集", "memory": "未采集",
            })
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
        if not url.startswith(("http://", "https://")):
            raise ToolExecutionError("健康检查地址仅支持 http 或 https")
        started = time.perf_counter()
        try:
            with urlopen(Request(url, method="GET"), timeout=self.timeout_seconds) as response:
                status = response.status
        except HTTPError as exc:
            # 4xx/5xx 是服务状态的重要证据，并不等同于采集工具失败。
            return {"status_code": exc.code, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
        except (URLError, ValueError, TimeoutError) as exc:
            raise ToolExecutionError(f"健康检查失败：{exc}") from exc
        return {"status_code": status, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

    def _git_recent_changes(self) -> Dict[str, Any]:
        raw = self._run(["git", "-C", str(self.repository), "log", "-1", "--format=%h%x00%s%x00%ci", "--name-only"])
        lines = raw.splitlines()
        header = lines[0].split("\x00") if lines else []
        if len(header) != 3:
            raise ToolExecutionError("无法解析最近一次 Git 提交")
        files = [line for line in lines[1:] if line]
        return {
            "summary": f"最近提交 {header[0]}：{header[1]}（{header[2]}）",
            "commits": [{"sha": header[0], "message": header[1], "files": files}],
        }

    def _config_diff(self) -> Dict[str, Any]:
        # 禁用外部 diff 与分页器，审查过程只读取 Git 原生文本，不执行仓库自定义程序。
        diff = self._run([
            "git", "--no-pager", "-c", "core.pager=cat", "-C", str(self.repository),
            "diff", "--no-ext-diff", "HEAD~1", "HEAD", "--",
        ])
        changes = []
        removed = re.findall(r"^-\s*batch_size\s*[:=]\s*(\d+)", diff, flags=re.MULTILINE)
        added = re.findall(r"^\+\s*batch_size\s*[:=]\s*(\d+)", diff, flags=re.MULTILINE)
        if removed and added:
            changes.append({"key": "batch_size", "before": int(removed[-1]), "after": int(added[-1])})
        return {"path": "由 Git 差异自动识别", "changes": changes, "raw_diff_available": bool(diff.strip())}

    def _git_code_review(self) -> Dict[str, Any]:
        """仅返回最新提交差异的有限片段，既不执行代码也不暴露整个仓库。"""
        diff = self._run([
            "git", "--no-pager", "-c", "core.pager=cat", "-C", str(self.repository),
            "diff", "--no-ext-diff", "HEAD~1", "HEAD", "--",
        ])
        limit = 20_000
        return {
            "base": "HEAD~1",
            "head": "HEAD",
            "diff": diff[:limit],
            "truncated": len(diff) > limit,
        }
