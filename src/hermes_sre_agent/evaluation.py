"""可重复的规则诊断评测：每个用例明确声明证据和预期结论。"""

import argparse
import json
from pathlib import Path

from .agent import SREAgent
from .backends import ScenarioBackend
from .triage import TriageAgent


DEFAULT_CASES = Path(__file__).resolve().parents[2] / "demo" / "evaluation_cases.json"


class CaseBackend:
    def __init__(self, case):
        self.base = ScenarioBackend(case["scenario"])
        self.overrides = case.get("overrides", {})
        self.fail_tools = set(case.get("fail_tools", []))

    def read(self, tool, **arguments):
        if tool in self.fail_tools:
            raise KeyError(f"评测用例模拟 {tool} 采集失败")
        return self.overrides[tool] if tool in self.overrides else self.base.read(tool, **arguments)


def evaluate_cases(path=DEFAULT_CASES, mode="rules"):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("评测文件必须是非空 JSON 用例数组")
    results = []
    for case in cases:
        agent = TriageAgent(CaseBackend(case)) if mode == "triage" else SREAgent(CaseBackend(case))
        run = agent.investigate(case["query"], case.get("service", "legal-agent"))
        diagnosis = run.diagnosis
        expected = case["expected"]
        checks = {
            "root_cause": expected["root_contains"] in diagnosis.root_cause,
            "confidence": diagnosis.confidence == expected["confidence"],
            "approval": diagnosis.needs_approval is expected["needs_approval"],
            "success": run.trajectory()["success"] is expected["success"],
        }
        results.append({"id": case["id"], "passed": all(checks.values()), "checks": checks,
                        "actual": {"root_cause": diagnosis.root_cause, "confidence": diagnosis.confidence,
                                   "needs_approval": diagnosis.needs_approval,
                                   "success": run.trajectory()["success"]}})
    return {"total": len(results), "passed": sum(result["passed"] for result in results), "cases": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hermes 可复现诊断评测")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="评测用例 JSON 文件")
    parser.add_argument("--mode", choices=("rules", "triage"), default="rules", help="评测规则诊断或多角色分诊")
    parser.add_argument("--output", type=Path, help="可选：将评测结果写入 JSON 文件")
    args = parser.parse_args(argv)
    try:
        result = evaluate_cases(args.cases, args.mode)
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        print(rendered, end="")
        return 0 if result["passed"] == result["total"] else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"评测未完成：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
