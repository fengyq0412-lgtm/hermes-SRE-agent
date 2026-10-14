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

    def test_model_cannot_change_service_health_target(self):
        model = FakeModelClient([
            json.dumps({"type": "tool_call", "tool": "service_health", "arguments": {"url": "http://other-host/health"}}),
            json.dumps({"type": "final", "diagnosis": {"evidence_tools": ["service_health"], "root_cause": "伪造结论"}}),
        ])
        run = LLMSREAgent(ScenarioBackend("api_slow"), model).investigate(
            "诊断服务", health_url="http://legal-agent:8000/health")
        self.assertIn("采集目标与本次诊断指定的服务不一致", run.calls[0].error)
        self.assertTrue(run.diagnosis.root_cause.startswith("未确定"))

    def test_wrong_tool_arguments_are_failed_calls_and_model_can_retry(self):
        model = FakeModelClient([
            json.dumps({"type": "tool_call", "tool": "system_metrics", "arguments": {"unexpected": 1}}),
            json.dumps({"type": "tool_call", "tool": "system_metrics", "arguments": {}}),
            json.dumps({"type": "final", "diagnosis": {
                "evidence_tools": ["system_metrics"], "root_cause": "未确定", "confidence": "低"}}),
        ])

        run = LLMSREAgent(ScenarioBackend("api_slow"), model).investigate("检查资源")

        self.assertEqual(len(run.calls), 2)
        self.assertIn("不接受参数", run.calls[0].error)
        self.assertIsNone(run.calls[1].error)
        self.assertEqual(run.diagnosis.root_cause, "未确定")
        self.assertTrue(any("请按工具契约重新调用" in message["content"] for message in model.messages[1]))

    def test_malformed_and_out_of_range_arguments_never_reach_backend(self):
        model = FakeModelClient([
            json.dumps({"type": "tool_call", "tool": "docker_logs", "arguments": []}),
            json.dumps({"type": "tool_call", "tool": "docker_logs",
                        "arguments": {"container": "legal-agent", "tail": True}}),
            json.dumps({"type": "tool_call", "tool": "docker_logs",
                        "arguments": {"container": "legal-agent", "tail": 5}}),
            json.dumps({"type": "final", "diagnosis": {
                "evidence_tools": ["docker_logs"], "root_cause": "未确定"}}),
        ])

        run = LLMSREAgent(ScenarioBackend("api_slow"), model).investigate("检查日志")

        self.assertEqual([call.error is None for call in run.calls], [False, False, True])
        self.assertIn("JSON 对象", run.calls[0].error)
        self.assertIn("1 到 200", run.calls[1].error)

    def test_cpu_metric_cannot_confirm_unrelated_database_root_cause(self):
        model = FakeModelClient([
            json.dumps({"type": "tool_call", "tool": "system_metrics", "arguments": {}}),
            json.dumps({"type": "final", "diagnosis": {
                "evidence_tools": ["system_metrics"], "observations": ["数据库已确认宕机"],
                "root_cause": "数据库宕机",
                "root_cause_signal": "database_unavailable", "confidence": "高",
                "remediation": ["立即重启数据库"], "needs_approval": True}}),
        ])

        run = LLMSREAgent(ScenarioBackend("api_slow"), model).investigate("服务为什么慢？")

        self.assertTrue(run.diagnosis.root_cause.startswith("未确定"))
        self.assertIn("数据库宕机", run.diagnosis.hypothesis)
        self.assertEqual(run.diagnosis.confidence, "低")
        self.assertFalse(run.diagnosis.needs_approval)
        self.assertNotIn("立即重启数据库", run.diagnosis.remediation)
        self.assertNotIn("数据库已确认宕机", run.diagnosis.observations)
        self.assertTrue(any("cpu_percent=" in item for item in run.diagnosis.observations))

    def test_confirmed_signal_requires_matching_observation_and_citation(self):
        class StoppedBackend:
            available_tools = ["docker_list", "system_metrics"]

            def read(self, tool, **_arguments):
                return ([{"name": "legal-agent", "status": "exited"}] if tool == "docker_list"
                        else {"cpu_percent": 99})

        for cited in (["system_metrics"], ["docker_list"]):
            with self.subTest(cited=cited):
                model = FakeModelClient([
                    json.dumps({"type": "tool_call", "tool": "system_metrics", "arguments": {}}),
                    json.dumps({"type": "tool_call", "tool": "docker_list", "arguments": {}}),
                    json.dumps({"type": "final", "diagnosis": {
                        "evidence_tools": cited, "root_cause": "数据库宕机",
                        "root_cause_signal": "container_stopped", "confidence": "高",
                        "remediation": ["立即重启数据库"]}}),
                ])
                run = LLMSREAgent(StoppedBackend(), model).investigate("故障诊断")
                if cited == ["docker_list"]:
                    self.assertIn("目标容器 legal-agent 未运行", run.diagnosis.root_cause)
                    self.assertIn("docker_list：legal-agent status=exited", run.diagnosis.evidence)
                    self.assertIn("数据库宕机", run.diagnosis.hypothesis)
                    self.assertNotIn("立即重启数据库", run.diagnosis.remediation)
                else:
                    self.assertTrue(run.diagnosis.root_cause.startswith("未确定"))

    def test_specific_signals_are_recomputed_from_tool_values(self):
        cases = (
            ("api_500", "missing_required_env", ("service_health", "docker_logs"), "MODEL_ENDPOINT", "高"),
            ("api_slow", "gpu_queue_pressure", ("service_health", "gpu_metrics", "docker_logs"),
             "GPU 显存压力", "中"),
        )
        for scenario, signal, tools, expected, confidence in cases:
            with self.subTest(signal=signal):
                arguments = {"service_health": {"url": "http://legal-agent:8000/health"},
                             "docker_logs": {"container": "legal-agent", "tail": 200}, "gpu_metrics": {}}
                replies = [json.dumps({"type": "tool_call", "tool": tool, "arguments": arguments[tool]})
                           for tool in tools]
                replies.append(json.dumps({"type": "final", "diagnosis": {
                    "evidence_tools": list(tools), "root_cause_signal": signal,
                    "root_cause": "模型自由文本结论", "confidence": "高"}}))

                run = LLMSREAgent(ScenarioBackend(scenario), FakeModelClient(replies)).investigate("诊断")

                self.assertIn(expected, run.diagnosis.root_cause)
                self.assertEqual(run.diagnosis.confidence, confidence)
                self.assertTrue(run.diagnosis.needs_approval)
