"""向 Docker 演示服务注入可恢复的故障状态，不执行任何宿主机破坏性操作。"""

import argparse
import json
from pathlib import Path
from typing import Dict


FAULTS: Dict[str, Dict] = {
    "normal": {
        "status_code": 200,
        "delay_seconds": 0.1,
        "batch_size": 16,
        "log_message": "INFO request completed normally batch_size=16",
    },
    "api_slow": {
        "status_code": 200,
        "delay_seconds": 4.2,
        "batch_size": 64,
        "log_message": "WARN reranker queue_wait_ms=3085 gpu_memory_pressure=true batch_size=64",
    },
    "api_500": {
        "status_code": 500,
        "delay_seconds": 0.05,
        "batch_size": 16,
        "log_message": "ERROR required environment variable MODEL_ENDPOINT is missing",
    },
}


def write_fault(scenario: str, runtime_dir: Path) -> Path:
    """原子替换状态文件，避免服务读取到只写了一半的 JSON。"""
    if scenario not in FAULTS:
        choices = "、".join(FAULTS)
        raise ValueError(f"未知故障场景：{scenario}。可用场景：{choices}")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    state = {"scenario": scenario, **FAULTS[scenario]}
    target = runtime_dir / "state.json"
    temporary = runtime_dir / "state.json.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="向 Hermes Docker 演示环境注入故障")
    parser.add_argument("--scenario", choices=sorted(FAULTS), required=True, help="要注入的故障场景")
    parser.add_argument("--runtime-dir", type=Path, default=Path(__file__).parent / "runtime", help="状态文件目录")
    args = parser.parse_args()
    path = write_fault(args.scenario, args.runtime_dir)
    print(f"已注入 {args.scenario}，状态文件：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
