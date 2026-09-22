import json
import unittest

from hermes_sre_agent.agent_mode import LLMSREAgent
from hermes_sre_agent.backends import ScenarioBackend


class FakeModelClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        return self.replies.pop(0)


class LlmAgentTests(unittest.TestCase):
    def test_model_selects_read_only_tool_then_returns_evidence_backed_diagnosis(self):
        model = FakeModelClient([
            json.dumps({"type": "tool_call", "tool": "service_health", "arguments": {"url": "http://legal-agent:8000/health"}}),
            json.dumps({
                "type": "final",
                "diagnosis": {
                    "observations": ["健康检查成功但延迟偏高。"],
                    "evidence_tools": ["service_health"],
                    "hypothesis": "服务处理变慢。",
                    "verification": "已读取健康检查延迟。",
                    "root_cause": "未确定",
                    "confidence": "低",
                    "remediation": ["继续采集日志。"],
                    "risk": "未执行操作。",
                    "needs_approval": False,
                },
            }),
        ])

        run = LLMSREAgent(ScenarioBackend("api_slow"), model).investigate("服务为什么慢？")

        self.assertEqual([call.tool for call in run.calls], ["service_health"])
        self.assertEqual(run.diagnosis.root_cause, "未确定")
        self.assertEqual(run.diagnosis.evidence, ["模型引用工具：service_health"])

    def test_model_cannot_claim_evidence_from_a_tool_that_never_succeeded(self):
        model = FakeModelClient([
            json.dumps({
                "type": "final",
                "diagnosis": {
                    "evidence_tools": ["docker_logs"],
                    "root_cause": "伪造结论",
                },
            }),
        ])

        run = LLMSREAgent(ScenarioBackend("api_slow"), model).investigate("服务为什么慢？")

        self.assertTrue(run.diagnosis.root_cause.startswith("未确定"))

    def test_model_cannot_bypass_approval_by_requesting_a_dangerous_tool(self):
        model = FakeModelClient([
            json.dumps({"type": "tool_call", "tool": "restart_service", "arguments": {"service": "legal-agent"}}),
            json.dumps({
                "type": "final",
                "diagnosis": {"evidence_tools": ["restart_service"], "root_cause": "伪造结论"},
            }),
        ])

        run = LLMSREAgent(ScenarioBackend("api_slow"), model).investigate("重启服务")

        self.assertEqual(run.calls[0].tool, "restart_service")
        self.assertIsNotNone(run.calls[0].error)
        self.assertTrue(run.diagnosis.root_cause.startswith("未确定"))

    def test_code_review_agent_cannot_expand_its_tool_scope(self):
        model = FakeModelClient([
            json.dumps({"type": "tool_call", "tool": "service_health", "arguments": {"url": "http://example.com"}}),
            json.dumps({"type": "final", "diagnosis": {"evidence_tools": ["service_health"], "root_cause": "伪造结论"}}),
        ])

        run = LLMSREAgent(
            ScenarioBackend("api_slow"), model,
            allowed_tools=["git_recent_changes", "config_diff", "git_code_review"],
        ).investigate("审查最近代码")

        self.assertEqual(run.calls[0].error, "当前 Agent 任务未获授权调用此工具。")
        self.assertTrue(run.diagnosis.root_cause.startswith("未确定"))
