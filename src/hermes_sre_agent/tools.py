"""受控工具层：只允许显式声明的能力，绝不提供任意 Shell。"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Union

from .backends import ScenarioBackend


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    read_only: bool
    requires_approval: bool = False


TOOL_SPECS = {
    "system_metrics": ToolSpec("system_metrics", "CPU、内存、磁盘与系统负载", True),
    "gpu_metrics": ToolSpec("gpu_metrics", "NVIDIA 使用率与显存", True),
    "docker_list": ToolSpec("docker_list", "容器状态与资源占用", True),
    "docker_logs": ToolSpec("docker_logs", "限定范围的最近容器日志", True),
    "service_health": ToolSpec("service_health", "HTTP 健康检查与延迟", True),
    "git_recent_changes": ToolSpec("git_recent_changes", "最近的部署相关变更", True),
    "config_diff": ToolSpec("config_diff", "当前配置与前一版本的差异", True),
    "git_code_review": ToolSpec("git_code_review", "限定长度的最近 Git 代码差异，供部署关联审查", True),
    "restart_service": ToolSpec("restart_service", "重启服务", False, True),
    "edit_config": ToolSpec("edit_config", "修改白名单配置文件", False, True),
}


class ToolExecutionError(RuntimeError):
    """只读采集失败时使用；调用方必须记录失败，不能伪造成功。"""


class ControlledTools:
    """只暴露声明过的工具；变更类工具在到达后端前即被拦截。"""

    def __init__(self, backend: ScenarioBackend):
        self.backend = backend

    def call(self, name: str, **arguments: Any) -> Union[Dict[str, Any], list]:
        spec = TOOL_SPECS.get(name)
        if spec is None:
            raise ValueError(f"工具 '{name}' 不在白名单中")
        if not spec.read_only:
            raise PermissionError(f"{name} 会改变系统状态，必须获得明确批准")
        return self.backend.read(name, **arguments)

    @staticmethod
    def list_specs() -> Iterable[ToolSpec]:
        return TOOL_SPECS.values()
