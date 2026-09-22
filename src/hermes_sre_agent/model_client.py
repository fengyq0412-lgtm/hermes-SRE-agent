"""兼容 OpenAI Chat Completions 协议的最小模型客户端。

不绑定单一供应商：只要服务实现了 `/v1/chat/completions`，即可通过环境变量接入。
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .env_loader import load_dotenv


class ModelConfigurationError(RuntimeError):
    """模型连接配置缺失或不安全时抛出。"""


class ModelRequestError(RuntimeError):
    """模型服务未能返回有效响应时抛出。"""


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 60

    @classmethod
    def from_environment(cls) -> "ModelConfig":
        load_dotenv()
        base_url = os.environ.get("HERMES_BASE_URL", "").rstrip("/")
        api_key = os.environ.get("HERMES_API_KEY", "")
        model = os.environ.get("HERMES_MODEL", "")
        if not all((base_url, api_key, model)):
            raise ModelConfigurationError(
                "Agent 模式需要 HERMES_BASE_URL、HERMES_API_KEY 和 HERMES_MODEL 三个环境变量。"
            )
        if not base_url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ModelConfigurationError("模型地址必须使用 HTTPS；本机 localhost / 127.0.0.1 可使用 HTTP。")
        return cls(base_url=base_url, api_key=api_key, model=model)


class OpenAICompatibleClient:
    """只使用标准库发起兼容请求，避免将项目绑定到特定 SDK。"""

    def __init__(self, config: ModelConfig):
        self.config = config

    def complete(self, messages: List[Dict[str, str]]) -> str:
        payload = json.dumps({
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.1,
        }, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.config.base_url}/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                data: Dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # 供应商错误体可能回显凭据或请求内容，不直接展示在网页。
            status = exc.code
            exc.close()
            raise ModelRequestError(f"模型服务返回 HTTP {status}，请检查 .env 中的地址、模型名、密钥或服务额度。") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelRequestError(f"模型请求失败：{exc}") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelRequestError("模型响应不符合 OpenAI Chat Completions 格式。") from exc
        if not isinstance(content, str):
            raise ModelRequestError("模型没有返回文本形式的 Agent 指令。")
        return content


def parse_json_reply(reply: str) -> Dict[str, Any]:
    """解析模型 JSON；兼容模型偶尔附带的 Markdown 代码围栏。"""
    text = reply.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0].strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelRequestError("模型没有返回约定的 JSON 指令。") from exc
    if not isinstance(result, dict):
        raise ModelRequestError("模型指令必须是 JSON 对象。")
    return result
