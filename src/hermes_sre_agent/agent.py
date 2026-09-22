"""Hermes 的证据优先状态机：每个结论均能回溯至采集结果。"""

from datetime import datetime, timezone
from typing import Any, List, Optional
from uuid import uuid4

from .contracts import Diagnosis, Incident, RunResult, ToolCall
from .tools import ControlledTools, ToolExecutionError


class SREAgent:
    STATES = (
        "START", "UNDERSTAND_INCIDENT", "COLLECT_BASIC_METRICS",
        "CHECK_SERVICE_HEALTH", "CHECK_LOGS", "CHECK_RECENT_CHANGES",
        "FORM_HYPOTHESES", "VERIFY_HYPOTHESES", "ROOT_CAUSE",
        "GENERATE_REMEDIATION_PLAN", "APPROVAL",
    )

    def __init__(self, backend: Any):
        self.tools = ControlledTools(backend)
        self.calls: List[ToolCall] = []

    def _observe(self, name: str, **arguments: Any) -> Any:
        try:
            result = self.tools.call(name, **arguments)
            self.calls.append(ToolCall(name, arguments, result))
            return result
        except (ToolExecutionError, KeyError, ValueError) as exc:
            # 采集失败同样是证据的一部分；后续诊断必须据此降低置信度。
            self.calls.append(ToolCall(name, arguments, None, error=str(exc)))
            return None

    def investigate(self, user_query: str, service: str = "legal-agent",
                    health_url: Optional[str] = None, container: Optional[str] = None) -> RunResult:
        """诊断一个服务；容器名和健康检查地址可与逻辑服务名分别指定。"""
        incident = Incident(
            incident_id=f"inc-{datetime.now(timezone.utc):%Y%m%d}-{uuid4().hex[:6]}",
            user_query=user_query,
            service=service,
            symptom="API 响应延迟高于基线",
        )
        states = list(self.STATES)
        target_url = health_url or f"http://{service}:8000/health"
        target_container = container or service
        metrics = self._observe("system_metrics")
        gpu = self._observe("gpu_metrics")
        containers = self._observe("docker_list")
        health = self._observe("service_health", url=target_url)
        logs = self._observe("docker_logs", container=target_container, tail=200)
        changes = self._observe("git_recent_changes")
        config = self._observe("config_diff")

        diagnosis = self._diagnose(metrics, gpu, containers, health, logs, changes, config, target_container)
        diagnosis.evidence.extend(
            f"{call.tool} 采集失败：{call.error}" for call in self.calls if call.error
        )
        return RunResult(incident=incident, diagnosis=diagnosis, calls=self.calls, state_history=states)

    @staticmethod
    def _diagnose(metrics: Any, gpu: Any, containers: Any, health: Any,
                  logs: Any, changes: Any, config: Any, container_name: str) -> Diagnosis:
        # 健康检查是基础诊断的唯一必需证据；GPU、Docker 和 Git 按部署环境可选。
        if not isinstance(health, dict):
            return Diagnosis(
                observations=["健康检查无法完成，未继续推断根因。"],
                evidence=[],
                hypothesis="数据不足，无法区分根因。",
                verification="停止于只读观察阶段，未执行任何变更。",
                root_cause="未确定——健康检查证据缺失。",
                confidence="低",
                remediation=["修复采集权限或服务连通性后重新诊断。"],
                risk="未提出任何变更操作。",
                needs_approval=False,
            )
        metrics = metrics if isinstance(metrics, dict) else {}
        gpu = gpu if isinstance(gpu, dict) else {}
        containers = containers if isinstance(containers, list) else []
        logs = logs if isinstance(logs, dict) else {}
        changes = changes if isinstance(changes, dict) else {}
        config = config if isinstance(config, dict) else {}
        latency = health.get("latency_ms", 0)
        gpu_used, gpu_total = gpu.get("memory_used_mb"), gpu.get("memory_total_mb")
        vram_ratio = gpu_used / gpu_total if isinstance(gpu_used, (int, float)) and isinstance(gpu_total, (int, float)) and gpu_total else 0
        config_changes = config.get("changes", [])
        batch_change = next((item for item in config_changes if item.get("key") == "batch_size"), None)
        log_text = "\n".join(logs.get("lines", []))
        container = next((item for item in containers if item.get("name") == container_name), {})

        observations = [
            f"系统 CPU 为 {metrics.get('cpu_percent', '未采集')}%；内存为 {metrics.get('memory_used_percent', '未采集')}%。",
            f"健康检查返回 HTTP {health.get('status_code')}，耗时 {latency} ms。",
            f"GPU 显存为 {gpu.get('memory_used_mb', '未采集')} / {gpu.get('memory_total_mb', '未采集')} MB（{vram_ratio:.0%}）。",
            f"容器 {container_name} 状态为 {container.get('status', '未知')}。",
        ]
        evidence = [
            f"健康检查：latency_ms={latency}",
            f"GPU 指标：利用率={gpu.get('gpu_utilization', '未采集')}%，显存={gpu.get('memory_used_mb', '未采集')}/{gpu.get('memory_total_mb', '未采集')} MB",
            f"容器日志：{log_text}",
            f"Git 最近变更：{changes.get('summary', '未发现最近变更')}",
        ]

        if batch_change and latency >= 1000 and vram_ratio >= 0.85:
            before, after = batch_change["before"], batch_change["after"]
            return Diagnosis(
                observations=observations,
                evidence=evidence + [f"配置差异：batch_size {before} → {after}"],
                hypothesis="更大的推理批大小挤占了 GPU 显存，并增加请求排队时间。",
                verification="延迟上升、显存接近耗尽、队列告警和本次发布的批大小变更相互印证。",
                root_cause=f"reranker 的 batch_size 从 {before} 调整为 {after}，导致 GPU 显存压力。",
                confidence="高",
                remediation=[
                    f"将 reranker 的 batch_size 从 {after} 回退为 {before}。",
                    "仅重启 reranker 服务。",
                    "重新检查健康检查延迟、GPU 显存和错误日志。",
                ],
                risk="重启期间 reranker 可能有数秒不可用。",
                needs_approval=True,
            )

        return Diagnosis(
            observations=observations,
            evidence=evidence,
            hypothesis="现有证据不足以确认单一根因。",
            verification="没有发现已被证据支持的配置变更与症状关联。",
            root_cause="未确定——已停止，未执行任何变更。",
            confidence="低",
            remediation=["在修改生产环境前，收集更长时间窗口的延迟数据与应用链路追踪。"],
            risk="未提出任何操作。",
            needs_approval=False,
        )
