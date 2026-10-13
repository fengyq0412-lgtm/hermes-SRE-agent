"""显式启用的只读 Kubernetes / Prometheus / Grafana 采集。"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .tools import ToolExecutionError


MAX_RESPONSE_BYTES = 128 * 1024


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


_OPENER = build_opener(_NoRedirect())


def _base_url(value: str) -> str:
    parsed = urlparse(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError("观测服务地址必须是无路径、凭据和查询参数的 http(s) 根地址")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("远程观测服务必须使用 HTTPS")
    return value.rstrip("/")


def _target(value: str, label: str) -> str:
    if not value or len(value) > 200 or value.startswith("-") or any(ord(char) < 32 for char in value):
        raise ValueError(f"{label} 不能为空、以 '-' 开头或包含控制字符，长度最多 200")
    return value


@dataclass(frozen=True)
class ObservabilityConfig:
    kube_context: Optional[str] = None
    kube_namespace: Optional[str] = None
    kube_selector: Optional[str] = None
    prometheus_url: Optional[str] = None
    prometheus_job: Optional[str] = None
    grafana_url: Optional[str] = None
    grafana_token_env: Optional[str] = None

    def __post_init__(self):
        if any((self.kube_context, self.kube_namespace, self.kube_selector)):
            if not all((self.kube_context, self.kube_namespace, self.kube_selector)):
                raise ValueError("Kubernetes 需要同时指定 context、namespace 和 selector")
            _target(self.kube_context, "Kubernetes context")
            if not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", self.kube_namespace) or len(self.kube_namespace) > 63:
                raise ValueError("Kubernetes namespace 格式无效")
            _target(self.kube_selector, "Kubernetes selector")
        if bool(self.prometheus_url) != bool(self.prometheus_job):
            raise ValueError("Prometheus 需要同时指定 URL 和 job")
        if self.prometheus_url:
            _base_url(self.prometheus_url)
            _target(self.prometheus_job, "Prometheus job")
        if bool(self.grafana_url) != bool(self.grafana_token_env):
            raise ValueError("Grafana 需要同时指定 URL 和令牌环境变量名")
        if self.grafana_url:
            _base_url(self.grafana_url)
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.grafana_token_env):
                raise ValueError("Grafana 令牌环境变量名格式无效")

    @property
    def enabled_tools(self):
        tools = []
        if self.kube_context:
            tools.append("kubernetes_workloads")
        if self.prometheus_url:
            tools.append("prometheus_scrape")
        if self.grafana_url:
            tools.append("grafana_alert_rules")
        return tools


def get_json(base_url: str, path: str, timeout: int, token: Optional[str] = None) -> Any:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with _OPENER.open(Request(base_url + path, headers=headers, method="GET"), timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise ToolExecutionError(f"观测接口返回 HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        # 不回显 URL、请求头或底层异常；其中可能含服务端提供的敏感内容。
        raise ToolExecutionError("观测接口连接失败或超时") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ToolExecutionError("观测接口响应超过 128 KiB 上限")
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ToolExecutionError("观测接口未返回有效 JSON") from exc


def prometheus_scrape(config: ObservabilityConfig, timeout: int) -> Dict[str, Any]:
    query = "up{job=" + json.dumps(config.prometheus_job, ensure_ascii=False) + "}"
    path = "/api/v1/query?" + urlencode({"query": query, "limit": "50"})
    payload = get_json(_base_url(config.prometheus_url), path, timeout)
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise ToolExecutionError("Prometheus 查询未成功")
    data = payload.get("data", {})
    if not isinstance(data, dict) or data.get("resultType") != "vector" or not isinstance(data.get("result"), list):
        raise ToolExecutionError("Prometheus 返回的瞬时向量格式无效")
    targets = []
    for item in data["result"][:50]:
        if not isinstance(item, dict) or not isinstance(item.get("metric"), dict):
            continue
        value = item.get("value", [])
        if not isinstance(value, list) or len(value) != 2:
            continue
        try:
            up = float(value[1]) == 1.0
        except (TypeError, ValueError):
            continue
        targets.append({"instance": str(item["metric"].get("instance", ""))[:200], "up": up})
    return {"job": config.prometheus_job, "targets": targets,
            "note": "up 仅表示 Prometheus 抓取是否成功；空列表表示没有匹配目标，不代表服务健康"}


def grafana_alert_rules(config: ObservabilityConfig, timeout: int) -> Dict[str, Any]:
    token = os.environ.get(config.grafana_token_env)
    if not token:
        raise ToolExecutionError("Grafana 令牌环境变量未设置")
    payload = get_json(_base_url(config.grafana_url), "/api/v1/provisioning/alert-rules", timeout, token)
    if not isinstance(payload, list):
        raise ToolExecutionError("Grafana 告警规则响应格式无效")
    rules = [{"uid": str(item.get("uid", ""))[:100], "title": str(item.get("title", ""))[:200],
              "isPaused": bool(item.get("isPaused", False))}
             for item in payload[:50] if isinstance(item, dict)]
    return {"rules": rules, "truncated": len(payload) > 50,
            "note": "此接口列出已配置规则，不表示规则当前正在告警"}
