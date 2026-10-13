"""V0.5 平台观测适配器的目标、输出和证据边界。"""

import io
import json
import os
import unittest
from unittest.mock import patch

from hermes_sre_agent.agent_mode import LLMSREAgent
from hermes_sre_agent.live_backend import LiveReadOnlyBackend
from hermes_sre_agent.observability import ObservabilityConfig
from hermes_sre_agent.tools import ToolExecutionError
from hermes_sre_agent.triage import TriageAgent


class _Response:
    def __init__(self, payload):
        self.stream = io.BytesIO(json.dumps(payload).encode())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stream.close()

    def read(self, size):
        return self.stream.read(size)


class ObservabilityTests(unittest.TestCase):
    def test_integrations_are_opt_in_and_targets_are_complete(self):
        backend = LiveReadOnlyBackend()
        self.assertNotIn("kubernetes_workloads", backend.available_tools)
        with self.assertRaisesRegex(ToolExecutionError, "不支持"):
            backend.read("kubernetes_workloads")
        with self.assertRaisesRegex(ValueError, "同时指定"):
            ObservabilityConfig(kube_context="production")
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            ObservabilityConfig(prometheus_url="http://remote.example", prometheus_job="app")
        with self.assertRaisesRegex(ValueError, "根地址"):
            ObservabilityConfig(grafana_url="https://grafana.example/other", grafana_token_env="GRAFANA_TOKEN")

    def test_kubernetes_runs_fixed_read_only_command_and_summarizes(self):
        config = ObservabilityConfig(kube_context="prod", kube_namespace="default", kube_selector="app=hermes")
        backend = LiveReadOnlyBackend(observability=config)
        captured = []
        backend._run = lambda command: captured.append(command) or json.dumps({"items": [
            {"kind": "Deployment", "metadata": {"name": "hermes"},
             "status": {"replicas": 2, "readyReplicas": 1, "secret": "omit"}},
        ]})
        result = backend.read("kubernetes_workloads")
        self.assertEqual(result["workloads"][0]["ready_replicas"], 1)
        self.assertNotIn("secret", str(result))
        self.assertEqual(captured[0], ["kubectl", "--context", "prod", "--namespace", "default",
                                       "--request-timeout", "5s", "get", "deployments,pods", "--selector",
                                       "app=hermes", "--output", "json"])
        with self.assertRaisesRegex(ToolExecutionError, "不能附加参数"):
            backend.read("kubernetes_workloads", namespace="other")

    def test_prometheus_uses_fixed_up_query_and_does_not_claim_app_health(self):
        config = ObservabilityConfig(prometheus_url="https://prom.example", prometheus_job='api"} or vector(1)')
        backend = LiveReadOnlyBackend(observability=config)
        payload = {"status": "success", "data": {"resultType": "vector", "result": [
            {"metric": {"instance": "a:9090", "private": "omit"}, "value": [123, "0"]}
        ]}}
        with patch("hermes_sre_agent.observability._OPENER.open", return_value=_Response(payload)) as opened:
            result = backend.read("prometheus_scrape")
        self.assertFalse(result["targets"][0]["up"])
        self.assertNotIn("private", str(result))
        self.assertIn("抓取", result["note"])
        requested = opened.call_args.args[0]
        self.assertIn("%5C%22", requested.full_url)
        self.assertEqual(requested.get_method(), "GET")

    def test_prometheus_accepts_numeric_one_point_zero(self):
        config = ObservabilityConfig(prometheus_url="https://prom.example", prometheus_job="hermes")
        payload = {"status": "success", "data": {"resultType": "vector", "result": [
            {"metric": {"instance": "a"}, "value": [123, "1.0"]}
        ]}}
        with patch("hermes_sre_agent.observability._OPENER.open", return_value=_Response(payload)):
            result = LiveReadOnlyBackend(observability=config).read("prometheus_scrape")
        self.assertTrue(result["targets"][0]["up"])

    def test_grafana_token_only_in_header_and_rule_data_excluded(self):
        config = ObservabilityConfig(grafana_url="https://grafana.example", grafana_token_env="TEST_GRAFANA_TOKEN")
        backend = LiveReadOnlyBackend(observability=config)
        payload = [{"uid": "one", "title": "High latency", "isPaused": False,
                    "data": [{"secret": "must-not-leak"}]}]
        with patch.dict(os.environ, {"TEST_GRAFANA_TOKEN": "very-secret"}), \
             patch("hermes_sre_agent.observability._OPENER.open", return_value=_Response(payload)) as opened:
            result = backend.read("grafana_alert_rules")
        self.assertEqual(result["rules"], [{"uid": "one", "title": "High latency", "isPaused": False}])
        self.assertNotIn("must-not-leak", str(result))
        self.assertNotIn("very-secret", opened.call_args.args[0].full_url)
        self.assertEqual(opened.call_args.args[0].get_header("Authorization"), "Bearer very-secret")

    def test_model_only_sees_enabled_tools_and_triage_marks_platform_evidence(self):
        config = ObservabilityConfig(kube_context="prod", kube_namespace="default", kube_selector="app=hermes")
        backend = LiveReadOnlyBackend(observability=config)
        agent = LLMSREAgent(backend, object())
        self.assertIn("kubernetes_workloads", agent.allowed_tools)
        self.assertNotIn("prometheus_scrape", agent.allowed_tools)
        backend._run = lambda command: json.dumps({"items": [
            {"kind": "Deployment", "metadata": {"name": "hermes"}, "status": {"replicas": 2, "readyReplicas": 1}}
        ]}) if command[0] == "kubectl" else ""
        backend._handlers.update({name: lambda **_: {} for name in (
            "service_health", "docker_list", "system_metrics", "gpu_metrics", "docker_logs",
            "git_recent_changes", "config_diff")})
        run = TriageAgent(backend).investigate("check", service="hermes")
        platform = next(item for item in run.assessments if item["role"] == "平台观测")
        self.assertEqual(platform["signal"], "degraded")
        self.assertEqual(platform["evidence_tools"], ["kubernetes_workloads"])


if __name__ == "__main__":
    unittest.main()
