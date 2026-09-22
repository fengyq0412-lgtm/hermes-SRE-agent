import json
import unittest

from hermes_sre_agent import SREAgent, ScenarioBackend
from hermes_sre_agent.reporting import render_report
from hermes_sre_agent.tools import ControlledTools


class AgentWorkflowTests(unittest.TestCase):
    def test_api_slow_creates_evidence_backed_diagnosis(self):
        run = SREAgent(ScenarioBackend("api_slow")).investigate("legal-agent 最近很慢，帮我排查原因")
        self.assertIn("batch_size 从 16 调整为 64", run.diagnosis.root_cause)
        self.assertEqual(run.diagnosis.confidence, "高")
        self.assertTrue(run.diagnosis.needs_approval)
        self.assertEqual([call.tool for call in run.calls], [
            "system_metrics", "gpu_metrics", "docker_list", "service_health",
            "docker_logs", "git_recent_changes", "config_diff",
        ])
        self.assertTrue(all(call.read_only for call in run.calls))
        self.assertIn("等待用户明确批准", render_report(run))
        self.assertEqual(json.loads(json.dumps(run.trajectory()))["actions"], [])

    def test_mutating_tool_cannot_run_without_approval(self):
        tools = ControlledTools(ScenarioBackend("api_slow"))
        with self.assertRaisesRegex(PermissionError, "必须获得明确批准"):
            tools.call("restart_service", service="qwen-reranker")

    def test_custom_health_url_and_container_are_used_for_collection(self):
        run = SREAgent(ScenarioBackend("api_slow")).investigate(
            "检查 Docker 演示环境", service="legal-agent", container="legal-agent",
            health_url="http://localhost:8000/health",
        )
        calls = {call.tool: call for call in run.calls}
        self.assertEqual(calls["service_health"].arguments["url"], "http://localhost:8000/health")
        self.assertEqual(calls["docker_logs"].arguments["container"], "legal-agent")

    def test_ambiguous_evidence_stops_safely(self):
        run = SREAgent(ScenarioBackend("api_500")).investigate("why is it failing?")
        self.assertTrue(run.diagnosis.root_cause.startswith("未确定"))
        self.assertFalse(run.diagnosis.needs_approval)

    def test_optional_collection_failure_does_not_hide_valid_health_evidence(self):
        class PartialBackend:
            def read(self, tool, **_):
                if tool == "service_health":
                    return {"status_code": 404, "latency_ms": 12}
                if tool == "system_metrics":
                    return {"cpu_percent": 20, "memory_used_percent": 40}
                raise KeyError(f"缺少可选数据：{tool}")

        run = SREAgent(PartialBackend()).investigate("服务是否正常")

        self.assertNotIn("健康检查证据缺失", run.diagnosis.root_cause)
        self.assertIn("HTTP 404", "\n".join(run.diagnosis.observations))
