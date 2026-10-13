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
        status_code = health.get("status_code")
        latency = SREAgent._number(health.get("latency_ms"))
        gpu_used, gpu_total = gpu.get("memory_used_mb"), gpu.get("memory_total_mb")
        gpu_used_number = SREAgent._number(gpu_used)
        gpu_total_number = SREAgent._number(gpu_total)
        vram_ratio = (gpu_used_number / gpu_total_number
                      if gpu_used_number is not None and gpu_total_number is not None and gpu_total_number > 0 else 0)
        config_changes = config.get("changes", [])
        batch_change = next((item for item in config_changes if item.get("key") == "batch_size"), None)
        log_text = "\n".join(logs.get("lines", []))
        container = next((item for item in containers if item.get("name") == container_name), {})
        container_status = str(container.get("status", "未知"))
        container_running = container_status.lower() == "running"
        cpu_percent = SREAgent._number(metrics.get("cpu_percent"))
        memory_percent = SREAgent._number(metrics.get("memory_used_percent"))

        observations = [
            f"系统 CPU 为 {metrics.get('cpu_percent', '未采集')}%；内存为 {metrics.get('memory_used_percent', '未采集')}%。",
            f"健康检查返回 HTTP {status_code if status_code is not None else '未采集'}，耗时 {latency if latency is not None else '未采集'} ms。",
            f"GPU 显存为 {gpu.get('memory_used_mb', '未采集')} / {gpu.get('memory_total_mb', '未采集')} MB（{vram_ratio:.0%}）。",
            f"容器 {container_name} 状态为 {container.get('status', '未知')}。",
        ]
        evidence = [
            f"健康检查：latency_ms={latency}",
            f"GPU 指标：利用率={gpu.get('gpu_utilization', '未采集')}%，显存={gpu.get('memory_used_mb', '未采集')}/{gpu.get('memory_total_mb', '未采集')} MB",
            f"容器日志：{log_text}",
            f"Git 最近变更：{changes.get('summary', '未发现最近变更')}",
        ]

        # 先判定可用性故障。HTTP 状态和容器生命周期均是直接观察结果，不能被后续的性能假设覆盖。
        if isinstance(status_code, int) and status_code >= 500:
            if container and not container_running:
                return Diagnosis(
                    observations=observations,
                    evidence=evidence + [f"容器状态：{container_name}={container_status}"],
                    hypothesis="服务健康检查失败且目标容器未处于运行状态。",
                    verification="HTTP 5xx 与容器非运行状态相互印证；尚未执行重启。",
                    root_cause=f"容器 {container_name} 当前状态为 {container_status}，服务不可用。",
                    confidence="高",
                    remediation=[
                        "检查容器退出码、启动日志和最近部署参数。",
                        f"确认原因后，申请批准再启动或回滚 {container_name}。",
                        "恢复后重新检查 HTTP 状态、延迟和错误日志。",
                    ],
                    risk="启动或回滚会改变服务状态，必须经人工批准。",
                    needs_approval=True,
                )

            missing = SREAgent._missing_environment_variable(log_text)
            if missing:
                return Diagnosis(
                    observations=observations,
                    evidence=evidence + [f"应用日志：缺少必需环境变量 {missing}"],
                    hypothesis="应用因必需环境变量缺失而无法完成请求处理。",
                    verification="健康检查返回 5xx，且应用日志明确报告缺少同一项启动配置。",
                    root_cause=f"服务缺少必需环境变量 {missing}，导致健康检查返回 HTTP {status_code}。",
                    confidence="高",
                    remediation=[
                        f"核对部署密钥或配置中 {missing} 的来源与作用域。",
                        "按变更流程补齐正确配置，避免在日志或命令行暴露值。",
                        "经人工批准后滚动发布，并重新检查健康状态与日志。",
                    ],
                    risk="补齐环境变量并发布会改变运行配置，必须经人工批准。",
                    needs_approval=True,
                )

            return Diagnosis(
                observations=observations,
                evidence=evidence,
                hypothesis="服务端返回 5xx，但现有日志没有可验证的具体异常原因。",
                verification="HTTP 状态证明服务失败；其余采集结果不足以归因到代码、配置或基础设施。",
                root_cause=f"未确定——健康检查返回 HTTP {status_code}，缺少可关联的服务端错误证据。",
                confidence="低",
                remediation=["扩大同一请求时间窗口内的应用错误日志和链路追踪后重新诊断。"],
                risk="未提出任何变更操作。",
                needs_approval=False,
            )

        if isinstance(status_code, int) and 400 <= status_code < 500:
            return Diagnosis(
                observations=observations,
                evidence=evidence,
                hypothesis="健康检查端点返回客户端错误，可能是地址、路由或访问策略不匹配。",
                verification="4xx 证明请求已到达服务边界，但不能单凭状态码确定是鉴权、路由还是探针配置问题。",
                root_cause=f"未确定——健康检查返回 HTTP {status_code}，需核对健康检查 URL、路由与访问策略。",
                confidence="低",
                remediation=["确认部署探针使用的 URL、认证要求及服务实际暴露的健康检查路由。"],
                risk="未提出任何变更操作。",
                needs_approval=False,
            )

        queue_evidence = "queue_wait" in log_text or "gpu_memory_pressure" in log_text
        if (batch_change and isinstance(status_code, int) and 200 <= status_code < 300
                and latency is not None and latency >= 1000 and vram_ratio >= 0.85 and queue_evidence):
            before, after = batch_change["before"], batch_change["after"]
            if config.get("source") == "repository_latest_commit" and not config.get("runtime_config_verified"):
                return Diagnosis(
                    observations=observations,
                    evidence=evidence + [f"仓库配置差异：{batch_change.get('path', '未知文件')} 的 batch_size {before} → {after}"],
                    hypothesis="仓库中的批大小变更可能与当前 GPU 压力和排队延迟有关。",
                    verification="仓库提交、GPU 指标和日志相符，但尚未确认该提交已部署及运行时配置值。",
                    root_cause="未确定——发现批大小变更线索，仍需核对实际部署版本与运行时配置。",
                    confidence="中",
                    remediation=["核对容器使用的镜像版本与运行时 batch_size，再决定是否回退。"],
                    risk="未提出变更操作。",
                    needs_approval=False,
                )
            return Diagnosis(
                observations=observations,
                evidence=evidence + [f"配置差异：{batch_change.get('path', config.get('path', '未知配置'))} 的 batch_size {before} → {after}"],
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

        if latency is not None and latency >= 1000 and cpu_percent is not None and cpu_percent >= 90:
            return Diagnosis(
                observations=observations,
                evidence=evidence + [f"系统指标：CPU={cpu_percent}%"],
                hypothesis="高延迟与主机 CPU 接近饱和同时出现，可能存在 CPU 竞争或并发过高。",
                verification="当前指标支持资源瓶颈假设，但缺少按进程、按时间窗口的 CPU 证据，不能确认唯一根因。",
                root_cause="未确定——检测到接口高延迟和主机 CPU 压力，需要进一步定位占用进程。",
                confidence="中",
                remediation=["采集服务进程和相邻容器的 CPU 使用率、请求并发数及排队时间。"],
                risk="未提出任何变更操作。",
                needs_approval=False,
            )

        if latency is not None and latency >= 1000 and memory_percent is not None and memory_percent >= 90:
            return Diagnosis(
                observations=observations,
                evidence=evidence + [f"系统指标：内存={memory_percent}%"],
                hypothesis="高延迟与主机内存压力同时出现，可能发生内存争用或回收。",
                verification="当前指标支持资源压力假设，但没有进程级内存、OOM 或回收日志，不能确认唯一根因。",
                root_cause="未确定——检测到接口高延迟和主机内存压力，需要进一步定位占用进程。",
                confidence="中",
                remediation=["采集进程级内存、内核 OOM/回收日志和容器内存限制后重新诊断。"],
                risk="未提出任何变更操作。",
                needs_approval=False,
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

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        """仅接受真实数值，避免把字符串、布尔值或缺失字段当作资源证据。"""
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    @staticmethod
    def _missing_environment_variable(log_text: str) -> Optional[str]:
        """从明确的错误日志中提取变量名，不猜测日志未出现的部署配置。"""
        marker = "missing required environment variable "
        position = log_text.lower().find(marker)
        if position < 0:
            return None
        candidate = log_text[position + len(marker):].split()[0] if log_text[position + len(marker):].split() else ""
        return candidate if candidate.replace("_", "").isalnum() and candidate.upper() == candidate else None
