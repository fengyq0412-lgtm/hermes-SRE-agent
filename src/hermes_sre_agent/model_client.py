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
    max_output_tokens: int = 4096

    @classmethod
    def from_environment(cls) -> "ModelConfig":
        load_dotenv()
        base_url = os.environ.get("HERMES_BASE_URL", "").rstrip("/")
        api_key = os.environ.get("HERMES_API_KEY", "")
        model = os.environ.get("HERMES_MODEL", "")
        try:
            max_output_tokens = int(os.environ.get("HERMES_MAX_OUTPUT_TOKENS", "4096"))
        except ValueError as exc:
            raise ModelConfigurationError("HERMES_MAX_OUTPUT_TOKENS 必须是 256 到 16384 的整数。") from exc
        if not 256 <= max_output_tokens <= 16384:
            raise ModelConfigurationError("HERMES_MAX_OUTPUT_TOKENS 必须是 256 到 16384 的整数。")
        if not all((base_url, api_key, model)):
            raise ModelConfigurationError(
                "Agent 模式需要 HERMES_BASE_URL、HERMES_API_KEY 和 HERMES_MODEL 三个环境变量。"
            )
        if not base_url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ModelConfigurationError("模型地址必须使用 HTTPS；本机 localhost / 127.0.0.1 可使用 HTTP。")
        return cls(base_url=base_url, api_key=api_key, model=model, max_output_tokens=max_output_tokens)


class OpenAICompatibleClient:
    """只使用标准库发起兼容请求，避免将项目绑定到特定 SDK。"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.last_usage = None
        self.last_finish_reason = None

    @staticmethod
    def _parse_usage(data):
        """仅接受供应商实际返回的非负整数用量，不估算字符数。"""
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
        total = usage.get("total_tokens")
        if not all(type(value) is int and value >= 0 for value in (prompt, completion)):
            return None
        if type(total) is not int or total < 0:
            total = prompt + completion
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}

    def complete(self, messages: List[Dict[str, str]]) -> str:
        self.last_usage = None
        self.last_finish_reason = None
        payload = json.dumps({
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": self.config.max_output_tokens,
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
        if not isinstance(data, dict):
            raise ModelRequestError("模型响应不符合 OpenAI Chat Completions 格式。")
        self.last_usage = self._parse_usage(data)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason in {"stop", "length", "content_filter", "tool_calls"}:
                self.last_finish_reason = finish_reason
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
