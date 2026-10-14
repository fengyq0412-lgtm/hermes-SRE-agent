"""网页版运行诊断：显式目标、只读采集、会话恢复与模型可选。"""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hermes_sre_agent.backends import ScenarioBackend
from hermes_sre_agent.web_server import HermesRequestHandler, ProjectRegistry, ReviewJobs, validate_diagnostic_target


def target(**changes):
    result = {"service": "hermes-web", "container": "hermes-web",
              "health_url": "http://127.0.0.1:8765/health", "strategy": "triage", "observability": {}}
    result.update(changes)
    return result


class FakeLiveBackend:
    configured = []

    def __init__(self, repository=None, observability=None):
        self.configured.append((Path(repository), observability))
        self.available_tools = ["system_metrics", "gpu_metrics", "docker_list", "docker_logs",
                                "service_health", "git_recent_changes", "config_diff", *observability.enabled_tools]
        self.scenario = ScenarioBackend("api_slow")

    def read(self, tool, **arguments):
        if tool == "kubernetes_workloads":
            return {"workloads": [{"kind": "Deployment", "name": "hermes-web", "replicas": 2, "ready_replicas": 1}]}
        if tool == "prometheus_scrape":
            return {"targets": [{"instance": "hermes-web", "up": False}]}
        if tool == "grafana_alert_rules":
            return {"rules": [{"uid": "latency", "title": "High latency"}]}
        return self.scenario.read(tool, **arguments)


class WebDiagnosisTests(unittest.TestCase):
    def test_target_rejects_missing_or_malformed_sources_before_queueing(self):
        for invalid in ({}, target(service=""), target(container="legal-agent; exit"),
                        target(health_url="file:///etc/passwd"),
                        target(health_url="http://127.0.0.1:invalid/health"),
                        target(observability={"prometheus_url": "https://prom.example"}),
                        target(observability={"grafana_token": "secret"})):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_diagnostic_target(invalid)
        with tempfile.TemporaryDirectory() as directory:
            registry = ProjectRegistry()
            project_id = registry.add(directory)["id"]
            jobs = ReviewJobs(registry)
            try:
                with self.assertRaises(ValueError):
                    jobs.create(project_id, "诊断", mode="diagnose", diagnostic_target={})
                self.assertEqual(jobs.jobs, {})
            finally:
                jobs.executor.shutdown(wait=True)

    def test_triage_collects_enabled_sources_without_model_and_restores_history(self):
        with tempfile.TemporaryDirectory() as directory, patch("hermes_sre_agent.web_server.LiveReadOnlyBackend", FakeLiveBackend):
            root = Path(directory)
            registry = ProjectRegistry()
            project_id = registry.add(directory)["id"]
            store = root / "conversations.sqlite3"
            jobs = ReviewJobs(registry, client_factory=lambda: self.fail("规则分诊不应调用模型"),
                              conversation_path=store)
            config = target(observability={
                "kube_context": "prod", "kube_namespace": "hermes", "kube_selector": "app=hermes-web",
                "prometheus_url": "https://prom.example", "prometheus_job": "hermes-web",
                "grafana_url": "https://grafana.example", "grafana_token_env": "GRAFANA_TOKEN",
            })
            try:
                created = jobs.create(project_id, "服务为什么变慢", mode="diagnose", diagnostic_target=config)
                jobs.futures[created["id"]].result(timeout=3)
                job = jobs.get(created["id"])
                self.assertEqual(job["status"], "completed")
                self.assertEqual(job["mode"], "diagnose")
                self.assertIn("平台观测", job["answer"])
                self.assertIn("运行诊断仅提供只读证据", job["answer"])
                self.assertNotIn("等待用户明确批准", job["answer"])
                self.assertEqual(len(job["diagnostic"]["tools"]), 10)
                self.assertEqual(job["usage"]["requests"], 0)
                self.assertEqual(FakeLiveBackend.configured[-1][0], root.resolve())
                self.assertEqual(FakeLiveBackend.configured[-1][1].kube_context, "prod")
                self.assertNotIn("secret", json.dumps(job))
            finally:
                jobs.executor.shutdown(wait=True)
            restored = ReviewJobs(registry, conversation_path=store)
            try:
                old = restored.get_conversation(project_id, created["conversation_id"])["turns"][0]
                self.assertEqual(old["status"], "completed")
                self.assertEqual(old["diagnostic_target"]["health_url"], config["health_url"])
                self.assertEqual(old["diagnostic"]["strategy"], "triage")
            finally:
                restored.executor.shutdown(wait=True)

    def test_model_assisted_diagnosis_uses_web_usage_meter(self):
        class Client:
            def __init__(self):
                self.calls = 0
                self.last_usage = None

            def complete(self, _messages):
                self.calls += 1
                self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                if self.calls == 1:
                    return json.dumps({"type": "tool_call", "tool": "service_health",
                                       "arguments": {"url": "http://127.0.0.1:8765/health"}})
                return json.dumps({"type": "final", "diagnosis": {"evidence_tools": ["service_health"],
                                   "root_cause": "未确定", "confidence": "低", "needs_approval": False}})

        with tempfile.TemporaryDirectory() as directory, patch("hermes_sre_agent.web_server.LiveReadOnlyBackend", FakeLiveBackend):
            registry = ProjectRegistry()
            project_id = registry.add(directory)["id"]
            jobs = ReviewJobs(registry, client_factory=Client)
            try:
                created = jobs.create(project_id, "检查服务", mode="diagnose",
                                      diagnostic_target=target(strategy="agent"))
                jobs.futures[created["id"]].result(timeout=3)
                job = jobs.get(created["id"])
                self.assertEqual(job["status"], "completed")
                self.assertEqual(job["usage"]["requests"], 2)
                self.assertEqual(job["diagnostic"]["tools"][0]["name"], "service_health")
                self.assertIsNone(job.get("proposal"))
            finally:
                jobs.executor.shutdown(wait=True)

    def test_web_model_diagnosis_records_bad_call_and_downgrades_unsupported_cause(self):
        class Client:
            def __init__(self):
                self.calls = 0
                self.last_usage = None

            def complete(self, _messages):
                self.calls += 1
                if self.calls == 1:
                    return json.dumps({"type": "tool_call", "tool": "system_metrics",
                                       "arguments": {"unexpected": "value"}})
                if self.calls == 2:
                    return json.dumps({"type": "tool_call", "tool": "system_metrics", "arguments": {}})
                return json.dumps({"type": "final", "diagnosis": {
                    "evidence_tools": ["system_metrics"], "root_cause": "数据库宕机",
                    "confidence": "高", "needs_approval": True}})

        with tempfile.TemporaryDirectory() as directory, patch("hermes_sre_agent.web_server.LiveReadOnlyBackend", FakeLiveBackend):
            registry = ProjectRegistry()
            project_id = registry.add(directory)["id"]
            jobs = ReviewJobs(registry, client_factory=Client)
            try:
                created = jobs.create(project_id, "数据库是否故障", mode="diagnose",
                                      diagnostic_target=target(strategy="agent"))
                jobs.futures[created["id"]].result(timeout=3)
                job = jobs.get(created["id"])
                self.assertEqual(job["status"], "completed")
                self.assertEqual([tool["ok"] for tool in job["diagnostic"]["tools"]], [False, True])
                self.assertIn("# 根因\n\n未确定", job["answer"])
                self.assertIn("待验证的根因推测：数据库宕机", job["answer"])
                self.assertIn("**置信度：** 低", job["answer"])
                self.assertNotIn("等待用户明确批准", job["answer"])
            finally:
                jobs.executor.shutdown(wait=True)

    def test_review_route_forwards_explicit_target(self):
        handler = object.__new__(HermesRequestHandler)
        handler.server = SimpleNamespace(server_port=8765)
        handler.path = "/api/reviews"
        payload = json.dumps({"project_id": "project-1", "question": "诊断", "mode": "diagnose",
                              "diagnostic_target": target()}).encode()
        handler.headers = {"Host": "127.0.0.1:8765", "X-Hermes-Token": handler.session_token,
                           "Content-Type": "application/json", "Content-Length": str(len(payload))}
        handler.rfile = io.BytesIO(payload)
        received = []
        handler.jobs = SimpleNamespace(create=lambda *args: received.append(args) or {"id": "job-1"})
        responses = []
        handler._send_json = lambda status, body: responses.append((status, body))
        handler.do_POST()
        self.assertEqual(responses, [(202, {"id": "job-1"})])
        self.assertEqual(received[0][-1], target())


if __name__ == "__main__":
    unittest.main()
