"""V0.4 三个只读分诊角色共享一次采集，由协调器合并当前证据。"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .agent import SREAgent
from .contracts import Incident, RunResult, ToolCall
from .tools import ControlledTools, ToolExecutionError


class TriageAgent:
    """固定职责的分诊角色；历史事件不会进入诊断判断。"""

    def __init__(self, backend: Any, on_observation=None):
        self.tools = ControlledTools(backend)
        self.on_observation = on_observation

    def investigate(self, user_query: str, service: str = "legal-agent",
                    health_url: Optional[str] = None, container: Optional[str] = None) -> RunResult:
        incident = Incident(
            incident_id=f"inc-{datetime.now(timezone.utc):%Y%m%d}-{uuid4().hex[:6]}",
            user_query=user_query, service=service,
        )
        target_url, target_container = health_url or f"http://{service}:8000/health", container or service
        requests = [
            ("service_health", {"url": target_url}),
            ("docker_list", {}),
            ("system_metrics", {}),
            ("gpu_metrics", {}),
            ("docker_logs", {"container": target_container, "tail": 200}),
            ("git_recent_changes", {}),
            ("config_diff", {}),
        ]
        available = set(getattr(self.tools.backend, "available_tools", ()))
        requests.extend((name, {}) for name in ("kubernetes_workloads", "prometheus_scrape", "grafana_alert_rules")
                        if name in available)
        calls: List[ToolCall] = []
        observed: Dict[str, Any] = {}
        for tool, arguments in requests:
            try:
                result = self.tools.call(tool, **arguments)
                calls.append(ToolCall(tool, arguments, result))
                observed[tool] = result
            except (ToolExecutionError, KeyError, ValueError) as exc:
                calls.append(ToolCall(tool, arguments, None, error=str(exc)))
                observed[tool] = None
            if self.on_observation:
                self.on_observation(calls[-1])

        availability = self._availability(observed, calls, target_container)
        performance = self._performance(observed, calls)
        changes = self._changes(observed, calls)
        assessments = [availability, performance, changes]
        if any(name in observed for name in ("kubernetes_workloads", "prometheus_scrape", "grafana_alert_rules")):
            assessments.append(self._observability(observed, calls))

        # 变更角色没有确认到批大小线索时，协调器不把原始 Git 输出交给根因规则。
        config = observed["config_diff"] if changes["signal"] in {"batch_change", "repository_candidate"} \
            and performance["signal"] == "gpu_pressure" else {"changes": []}
        diagnosis = SREAgent._diagnose(
            observed["system_metrics"], observed["gpu_metrics"], observed["docker_list"],
            observed["service_health"], observed["docker_logs"], observed["git_recent_changes"],
            config, target_container,
        )
        diagnosis.evidence.extend(f"{call.tool} 采集失败：{call.error}" for call in calls if call.error)

        if availability["signal"] in {"container_stopped", "http_5xx", "http_4xx"}:
            decision = "优先处理可用性证据"
        elif performance["signal"] == "gpu_pressure" and changes["signal"] in {"batch_change", "repository_candidate"}:
            decision = "交叉核对性能与配置变更"
        else:
            decision = "保留未证实线索并继续定位"
        assessments.append({"role": "协调器", "signal": "decision", "summary": decision,
                            "evidence_tools": sorted({tool for item in assessments for tool in item["evidence_tools"]})})
        return RunResult(
            incident, diagnosis, calls,
            ["START", "TRIAGE_COLLECT", "ASSESS_AVAILABILITY", "ASSESS_PERFORMANCE",
             "ASSESS_CHANGES", "COORDINATE", "APPROVAL"], assessments=assessments,
        )

    @staticmethod
    def _evidence(calls: List[ToolCall], names) -> List[str]:
        return [call.tool for call in calls if call.tool in names and call.error is None]

    @classmethod
    def _availability(cls, observed, calls, container_name):
        health = observed["service_health"] if isinstance(observed["service_health"], dict) else {}
        containers = observed["docker_list"] if isinstance(observed["docker_list"], list) else []
        target = next((item for item in containers if isinstance(item, dict) and item.get("name") == container_name), {})
        status = health.get("status_code")
        if type(status) is not int:
            signal, summary = "unknown", "健康检查证据缺失"
        elif status >= 500 and target and target.get("status") != "running":
            signal, summary = "container_stopped", "HTTP 5xx 且目标容器未运行"
        elif status >= 500:
            signal, summary = "http_5xx", "服务返回 HTTP 5xx"
        elif status >= 400:
            signal, summary = "http_4xx", "健康检查返回 HTTP 4xx"
        elif 200 <= status < 300:
            signal, summary = "available", "健康检查返回成功状态"
        else:
            signal, summary = "unexpected_http", "健康检查返回非预期状态"
        return {"role": "可用性", "signal": signal, "summary": summary,
                "evidence_tools": cls._evidence(calls, {"service_health", "docker_list"})}

    @classmethod
    def _performance(cls, observed, calls):
        health = observed["service_health"] if isinstance(observed["service_health"], dict) else {}
        metrics = observed["system_metrics"] if isinstance(observed["system_metrics"], dict) else {}
        gpu = observed["gpu_metrics"] if isinstance(observed["gpu_metrics"], dict) else {}
        logs = observed["docker_logs"] if isinstance(observed["docker_logs"], dict) else {}
        latency = SREAgent._number(health.get("latency_ms"))
        used, total = SREAgent._number(gpu.get("memory_used_mb")), SREAgent._number(gpu.get("memory_total_mb"))
        log_text = "\n".join(str(item) for item in logs.get("lines", []))
        if (latency is not None and latency >= 1000 and used is not None and total and used / total >= 0.85
                and ("queue_wait" in log_text or "gpu_memory_pressure" in log_text)):
            signal, summary = "gpu_pressure", "高延迟、GPU 显存压力和排队日志相符"
        elif latency is not None and latency >= 1000 and (SREAgent._number(metrics.get("cpu_percent")) or 0) >= 90:
            signal, summary = "cpu_pressure", "高延迟与主机 CPU 压力同时出现"
        elif latency is not None and latency >= 1000 and (SREAgent._number(metrics.get("memory_used_percent")) or 0) >= 90:
            signal, summary = "memory_pressure", "高延迟与主机内存压力同时出现"
        elif latency is not None and latency >= 1000:
            signal, summary = "slow_unknown", "接口延迟高，但资源证据不足"
        else:
            signal, summary = "no_pressure_evidence", "未发现可确认的性能压力证据"
        return {"role": "性能", "signal": signal, "summary": summary,
                "evidence_tools": cls._evidence(calls, {"service_health", "system_metrics", "gpu_metrics", "docker_logs"})}

    @classmethod
    def _changes(cls, observed, calls):
        config = observed["config_diff"] if isinstance(observed["config_diff"], dict) else {}
        changes = config.get("changes", [])
        batch = next((item for item in changes if isinstance(item, dict) and item.get("key") == "batch_size"), None) \
            if isinstance(changes, list) else None
        if batch and config.get("source") == "repository_latest_commit" and not config.get("runtime_config_verified"):
            signal, summary = "repository_candidate", "仓库有批大小变更，尚未核对运行时配置"
        elif batch:
            signal, summary = "batch_change", "发现批大小配置变更"
        else:
            signal, summary = "no_batch_change", "未发现可用的批大小变更证据"
        return {"role": "变更", "signal": signal, "summary": summary,
                "evidence_tools": cls._evidence(calls, {"git_recent_changes", "config_diff"})}

    @classmethod
    def _observability(cls, observed, calls):
        kube = observed.get("kubernetes_workloads") or {}
        prom = observed.get("prometheus_scrape") or {}
        grafana = observed.get("grafana_alert_rules") or {}
        workloads = kube.get("workloads", []) if isinstance(kube, dict) else []
        targets = prom.get("targets", []) if isinstance(prom, dict) else []
        rules = grafana.get("rules", []) if isinstance(grafana, dict) else []
        unready = [item for item in workloads if item.get("kind") == "Deployment"
                   and isinstance(item.get("replicas"), int)
                   and (item.get("ready_replicas") or 0) < item["replicas"]]
        unhealthy_pods = [item for item in workloads if item.get("kind") == "Pod"
                          and item.get("phase") in {"Pending", "Failed", "Unknown"}]
        failed = [item for item in targets if item.get("up") is False]
        if unready or unhealthy_pods or failed:
            signal, summary = "degraded", (f"发现 {len(unready)} 个副本不足的 Deployment、"
                                            f"{len(unhealthy_pods)} 个状态异常的 Pod、{len(failed)} 个抓取失败目标；"
                                            "需与服务故障交叉验证")
        elif workloads or targets or rules:
            signal, summary = "snapshot", "已收集平台观测快照；仅凭配置规则或抓取成功不能证明服务健康"
        else:
            signal, summary = "no_evidence", "平台观测数据为空或采集失败"
        names = {"kubernetes_workloads", "prometheus_scrape", "grafana_alert_rules"}
        return {"role": "平台观测", "signal": signal, "summary": summary,
                "evidence_tools": cls._evidence(calls, names)}
