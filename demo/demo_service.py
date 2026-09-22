"""用于演示的极简 HTTP 服务：通过状态文件在无需重建容器的情况下切换故障。"""

import json
import os
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict


STATE_PATH = Path(os.environ.get("HERMES_DEMO_STATE", "/app/runtime/state.json"))
LOG_PATH = Path(os.environ.get("HERMES_DEMO_LOG", "/app/logs/legal-agent.log"))

DEFAULT_STATE = {
    "scenario": "normal",
    "status_code": 200,
    "delay_seconds": 0.1,
    "batch_size": 16,
    "log_message": "INFO request completed normally",
}


def load_state() -> Dict[str, Any]:
    """状态文件不存在或损坏时回落到健康状态，确保服务本身可用。"""
    try:
        content = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_STATE.copy()
    return {**DEFAULT_STATE, **content}


def append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} {message}\n")


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - HTTP 框架要求固定方法名
        if self.path != "/health":
            self.send_error(404, "只提供 /health 端点")
            return
        state = load_state()
        time.sleep(float(state["delay_seconds"]))
        status = int(state["status_code"])
        append_log(str(state["log_message"]))
        body = json.dumps({
            "service": "legal-agent",
            "scenario": state["scenario"],
            "batch_size": state["batch_size"],
            "status": "ok" if status == 200 else "error",
        }, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: Any) -> None:
        """避免 HTTP 框架重复写日志，演示日志统一由 append_log 输出。"""


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8000), DemoHandler)
    print("Hermes 演示服务已在 http://0.0.0.0:8000/health 启动", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
