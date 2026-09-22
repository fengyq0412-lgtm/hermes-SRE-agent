import json
from pathlib import Path

from .contracts import RunResult


def render_report(run: RunResult) -> str:
    d, i = run.diagnosis, run.incident
    lines = [
        "# 事件摘要", "", f"**事件：** `{i.incident_id}`", f"**服务：** {i.service}",
        f"**症状：** {i.symptom}", f"**严重性：** {'中等' if d.confidence == '高' else '需要人工研判'}", "",
        "# 观察结果", "", *[f"- {item}" for item in d.observations], "",
        "# 证据", "", *[f"- {item}" for item in d.evidence], "",
        "# 假设与验证", "", d.hypothesis, "", d.verification, "",
        "# 根因", "", f"{d.root_cause}", f"\n**置信度：** {d.confidence}", "",
        "# 建议修复方案", "", *[f"{idx}. {item}" for idx, item in enumerate(d.remediation, 1)], "",
        "# 风险", "", d.risk, "",
        "# 审批", "", "等待用户明确批准；尚未执行任何会改变系统状态的命令。"
        if d.needs_approval else "未提出操作；Hermes 已安全停止。",
    ]
    return "\n".join(lines) + "\n"


def write_artifacts(run: RunResult, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / f"{run.incident.incident_id}-report.md"
    trajectory_path = directory / f"{run.incident.incident_id}-trajectory.json"
    report_path.write_text(render_report(run), encoding="utf-8")
    trajectory_path.write_text(json.dumps(run.trajectory(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_path, trajectory_path
