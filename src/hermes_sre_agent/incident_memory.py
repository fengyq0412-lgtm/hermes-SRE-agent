"""事件索引只记录结构化症状和报告位置，不保存原始提问、日志或工具输出。"""

import os
import sqlite3
import stat
from datetime import datetime, timezone
from pathlib import Path

from .agent import SREAgent
from .contracts import RunResult


def incident_signature(run: RunResult):
    results = {call.tool: call.result for call in run.calls if call.error is None}
    health = results.get("service_health") if isinstance(results.get("service_health"), dict) else {}
    containers = results.get("docker_list") if isinstance(results.get("docker_list"), list) else []
    logs = results.get("docker_logs") if isinstance(results.get("docker_logs"), dict) else {}
    gpu = results.get("gpu_metrics") if isinstance(results.get("gpu_metrics"), dict) else {}
    config = results.get("config_diff") if isinstance(results.get("config_diff"), dict) else {}
    status = health.get("status_code")
    status_class = f"{status // 100}xx" if type(status) is int and 100 <= status <= 599 else "unknown"
    latency = SREAgent._number(health.get("latency_ms"))
    slow = latency is not None and latency >= 1000
    target_name = next((call.arguments.get("container") for call in run.calls if call.tool == "docker_logs"), run.incident.service)
    target = next((item for item in containers if isinstance(item, dict) and item.get("name") == target_name), {})
    stopped = bool(target and target.get("status") != "running")
    log_text = "\n".join(str(line) for line in logs.get("lines", []))
    missing_env = bool(SREAgent._missing_environment_variable(log_text))
    used, total = SREAgent._number(gpu.get("memory_used_mb")), SREAgent._number(gpu.get("memory_total_mb"))
    gpu_pressure = bool(used is not None and total and used / total >= 0.85)
    changes = config.get("changes", [])
    batch_change = isinstance(changes, list) and any(isinstance(item, dict) and item.get("key") == "batch_size" for item in changes)

    if status_class == "5xx" and stopped:
        category = "容器不可用"
    elif status_class == "5xx" and missing_env:
        category = "配置缺失与 HTTP 5xx"
    elif status_class == "5xx":
        category = "HTTP 5xx"
    elif status_class == "4xx":
        category = "HTTP 4xx"
    elif slow and gpu_pressure and batch_change:
        category = "慢接口与 GPU/配置线索"
    elif slow:
        category = "慢接口"
    elif status_class == "unknown":
        category = "健康检查证据缺失"
    else:
        category = "常规观察"
    fingerprint = "|".join((status_class, "slow" if slow else "normal", "stopped" if stopped else "running-or-unknown",
                            "missing-env" if missing_env else "no-env-error",
                            "gpu-pressure" if gpu_pressure else "no-gpu-pressure",
                            "batch-change" if batch_change else "no-batch-change"))
    return fingerprint, category


class IncidentMemory:
    def __init__(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "incidents.sqlite3"
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            info = self.path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("事件记忆文件不是普通文件")
            os.chmod(self.path, 0o600)
        else:
            os.close(fd)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS incidents ("
                       "id TEXT PRIMARY KEY, service TEXT NOT NULL, source TEXT NOT NULL, fingerprint TEXT NOT NULL, "
                       "category TEXT NOT NULL, outcome TEXT NOT NULL, confidence TEXT NOT NULL, "
                       "created_at TEXT NOT NULL, report_path TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS incidents_related ON incidents(service, source, fingerprint, created_at DESC)")

    @staticmethod
    def _row(row):
        return {"incident_id": row[0], "service": row[1], "source": row[2], "category": row[3],
                "outcome": row[4], "confidence": row[5], "created_at": row[6], "report_path": row[7]}

    def find_related(self, run: RunResult, source: str, limit: int = 3):
        fingerprint, _ = incident_signature(run)
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT id, service, source, category, outcome, confidence, created_at, report_path "
                              "FROM incidents WHERE service=? AND source=? AND fingerprint=? AND id<>? "
                              "ORDER BY created_at DESC LIMIT ?",
                              (run.incident.service, source, fingerprint, run.incident.incident_id, min(max(limit, 1), 10))).fetchall()
        return [self._row(row) for row in rows]

    def record(self, run: RunResult, report_path: Path, source: str):
        fingerprint, category = incident_signature(run)
        outcome = "有诊断结论" if run.trajectory()["success"] else "根因待确认"
        created_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO incidents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (run.incident.incident_id, run.incident.service, source, fingerprint, category,
                        outcome, run.diagnosis.confidence, created_at, str(Path(report_path).resolve())))

    def history(self, service: str, source: str, limit: int = 20):
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT id, service, source, category, outcome, confidence, created_at, report_path "
                              "FROM incidents WHERE service=? AND source=? ORDER BY created_at DESC LIMIT ?",
                              (service, source, min(max(limit, 1), 100))).fetchall()
        return [self._row(row) for row in rows]
