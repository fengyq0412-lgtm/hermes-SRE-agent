"""可移植、确定性的演示数据源，保证每次演示都能得到同一条证据链。"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class ScenarioBackend:
    def __init__(self, scenario: str = "api_slow", root: Optional[Path] = None):
        base = root or Path(__file__).resolve().parents[2] / "demo" / "scenarios"
        path = base / f"{scenario}.json"
        if not path.exists():
            choices = ", ".join(p.stem for p in base.glob("*.json"))
            raise ValueError(f"未知演示场景 '{scenario}'。可用场景：{choices}")
        self.name = scenario
        self.data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    def read(self, tool: str, **_: Any) -> Any:
        if tool not in self.data:
            raise KeyError(f"演示场景 '{self.name}' 未提供工具 {tool} 的数据")
        return self.data[tool]
