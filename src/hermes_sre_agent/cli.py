import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Callable

from .agent import SREAgent
from .agent_mode import LLMSREAgent
from .backends import ScenarioBackend
from .env_loader import load_dotenv
from .incident_memory import IncidentMemory
from .live_backend import LiveReadOnlyBackend
from .model_client import ModelConfig, ModelConfigurationError, OpenAICompatibleClient
from .observability import ObservabilityConfig
from .reporting import render_report, write_artifacts
from .tools import ControlledTools
from .triage import TriageAgent


def _require_live_target(service: str, health_url: str = None, container: str = None,
                         history: bool = False) -> None:
    """实时采集不能沿用演示服务、容器或探针的隐式默认值。"""
    targets = {"--service": service}
    if not history:
        targets.update({"--container": container, "--health-url": health_url})
    missing = [name for name, value in targets.items() if not isinstance(value, str) or not value.strip()]
    if missing:
        raise ValueError("--backend live 必须明确指定 " + "、".join(missing) + "；不会使用演示目标作为默认值")


def run_diagnosis(query: str, backend_name: str, scenario: str, service: str, output_dir: Path,
                  health_url: str = None, container: str = None, mode: str = "agent",
                  remember: bool = True, observability: ObservabilityConfig = None) -> None:
    """执行一次独立诊断，并持久化人类报告和评测轨迹。"""
    if backend_name == "live":
        _require_live_target(service, health_url, container)
    backend = ScenarioBackend(scenario) if backend_name == "scenario" else LiveReadOnlyBackend(observability=observability)
    if mode == "rules":
        run = SREAgent(backend).investigate(query, service, health_url, container)
    elif mode == "triage":
        run = TriageAgent(backend).investigate(query, service, health_url, container)
    else:
        run = LLMSREAgent(backend, OpenAICompatibleClient(ModelConfig.from_environment())).investigate(
            query, service, health_url, container,
        )
    memory = None
    if remember:
        try:
            memory = IncidentMemory(output_dir)
            run.related_incidents = memory.find_related(run, backend_name)
        except (OSError, sqlite3.Error, ValueError) as exc:
            run.memory_error = str(exc)
    report, trajectory = write_artifacts(run, output_dir)
    if memory:
        try:
            memory.record(run, report, backend_name)
        except (OSError, sqlite3.Error, ValueError) as exc:
            run.memory_error = str(exc)
            report, trajectory = write_artifacts(run, output_dir)
    print(render_report(run))
    print(f"\n生成文件：{report} | {trajectory}")


def run_interactive(backend_name: str, scenario: str, service: str, output_dir: Path,
                    health_url: str = None, container: str = None,
                    mode: str = "agent", read_input: Callable[[str], str] = input,
                    remember: bool = True, observability: ObservabilityConfig = None) -> None:
    """提供最小化终端交互：每个提问都是一条独立、可追溯的事件。"""
    if backend_name == "live":
        _require_live_target(service, health_url, container)
    print("Hermes SRE Agent 交互模式。输入故障描述开始诊断；输入 exit 或 quit 退出。")
    while True:
        try:
            query = read_input("\nhermes> ").strip()
        except EOFError:
            print("\n已退出 Hermes。")
            return
        if query.lower() in {"exit", "quit"}:
            print("已退出 Hermes。")
            return
        if not query:
            continue
        try:
            run_diagnosis(query, backend_name, scenario, service, output_dir, health_url, container,
                          mode, remember, observability)
        except ModelConfigurationError as exc:
            print(f"Agent 模式尚未配置：{exc}")
            print("可先使用 --mode rules 运行确定性演示，或配置兼容模型后重试。")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="证据优先、审批受控的 SRE 诊断助手")
    parser.add_argument("query", nargs="?", help="故障描述，例如：服务为什么变慢了？")
    parser.add_argument("--backend", choices=("scenario", "live"), default="scenario", help="数据来源：演示场景或本机只读采集")
    parser.add_argument("--mode", choices=("agent", "rules", "triage"), default="agent",
                        help="agent 由模型选工具；rules 为规则演示；triage 为三角色只读分诊")
    parser.add_argument("--scenario", default="api_slow", help="确定性的演示场景")
    parser.add_argument("--service", help="目标服务名；实时模式必填，演示模式默认 legal-agent")
    parser.add_argument("--container", help="Docker 容器名；实时模式必填，演示模式默认与服务名相同")
    parser.add_argument("--health-url", help="健康检查 URL；实时模式必填，演示模式默认 http://<服务名>:8000/health")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="报告、轨迹和事件记忆目录")
    parser.add_argument("--list-tools", action="store_true", help="列出受控工具白名单")
    parser.add_argument("--interactive", action="store_true", help="进入连续交互式终端")
    parser.add_argument("--history", action="store_true", help="查看该服务最近的事件索引，不调用模型或采集工具")
    parser.add_argument("--no-memory", action="store_true", help="本次不读取或记录事件记忆")
    parser.add_argument("--kube-context", help="显式指定 Kubernetes context")
    parser.add_argument("--kube-namespace", help="显式指定 Kubernetes namespace")
    parser.add_argument("--kube-selector", help="限定工作负载的 label selector")
    parser.add_argument("--prometheus-url", help="Prometheus 根地址；远程地址必须为 HTTPS")
    parser.add_argument("--prometheus-job", help="查询 up 指标时匹配的 job 标签")
    parser.add_argument("--grafana-url", help="Grafana 根地址；远程地址必须为 HTTPS")
    parser.add_argument("--grafana-token-env", help="存放 Grafana 服务账号令牌的环境变量名，不传令牌原文")
    args = parser.parse_args()
    if args.list_tools:
        for spec in ControlledTools.list_specs():
            access = "只读" if spec.read_only else "需审批"
            print(f"{spec.name:20} {access:20} {spec.description}")
        return 0
    if args.backend == "live":
        try:
            _require_live_target(args.service, args.health_url, args.container, history=args.history)
        except ValueError as exc:
            parser.error(str(exc))
    service = args.service if args.service is not None else "legal-agent"
    if args.history:
        try:
            print(json.dumps(IncidentMemory(args.output_dir).history(service, args.backend), ensure_ascii=False, indent=2))
        except (OSError, sqlite3.Error, ValueError) as exc:
            print(f"无法读取事件记忆：{exc}", file=sys.stderr)
            return 2
        return 0
    try:
        observability = ObservabilityConfig(
            args.kube_context, args.kube_namespace, args.kube_selector,
            args.prometheus_url, args.prometheus_job, args.grafana_url, args.grafana_token_env,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if observability.enabled_tools and args.backend != "live":
        parser.error("平台观测集成仅支持 --backend live")
    if observability.enabled_tools and args.mode == "rules":
        parser.error("平台观测集成仅支持 --mode agent 或 --mode triage")
    if args.interactive:
        run_interactive(args.backend, args.scenario, service, args.output_dir, args.health_url,
                        args.container, args.mode, remember=not args.no_memory, observability=observability)
        return 0
    if not args.query:
        parser.error("必须提供故障描述")
    try:
        run_diagnosis(args.query, args.backend, args.scenario, service, args.output_dir,
                      args.health_url, args.container, args.mode, not args.no_memory, observability)
    except ModelConfigurationError as exc:
        parser.error(f"Agent 模式尚未配置：{exc} 可使用 --mode rules 运行确定性演示。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
