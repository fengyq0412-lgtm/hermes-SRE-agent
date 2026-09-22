from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    tool: str
    arguments: Dict[str, Any]
    result: Any
    read_only: bool = True
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Incident:
    incident_id: str
    user_query: str
    service: str = "legal-agent"
    symptom: str = "接口响应延迟高于基线"


@dataclass
class Diagnosis:
    observations: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    hypothesis: str = ""
    verification: str = ""
    root_cause: str = ""
    confidence: str = "Low"
    remediation: List[str] = field(default_factory=list)
    risk: str = ""
    needs_approval: bool = False


@dataclass
class RunResult:
    incident: Incident
    diagnosis: Diagnosis
    calls: List[ToolCall]
    state_history: List[str]

    def trajectory(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident.incident_id,
            "user_query": self.incident.user_query,
            "steps": [call.as_dict() for call in self.calls],
            "root_cause": self.diagnosis.root_cause,
            "actions": [],
            "success": bool(self.diagnosis.root_cause),
            "state_history": self.state_history,
        }
