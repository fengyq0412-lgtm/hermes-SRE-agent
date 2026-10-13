import unittest
from subprocess import CompletedProcess
from urllib.error import HTTPError
from unittest.mock import patch

from hermes_sre_agent.live_backend import LiveReadOnlyBackend
from hermes_sre_agent.tools import ToolExecutionError


class LiveBackendSafetyTests(unittest.TestCase):
    def test_illegal_container_name_is_rejected_before_command_execution(self):
        backend = LiveReadOnlyBackend()
        with self.assertRaisesRegex(ToolExecutionError, "容器名仅允许"):
            backend._docker_logs("legal-agent; rm -rf /", 20)

    def test_log_tail_is_bounded_and_passed_as_an_argument(self):
        backend = LiveReadOnlyBackend()
        captured = []
        backend._run = lambda command: captured.append(command) or "one\ntwo"

        result = backend._docker_logs("legal-agent", 9999)

        self.assertEqual(result, {"lines": ["one", "two"]})
        self.assertEqual(captured, [["docker", "logs", "--tail", "200", "--", "legal-agent"]])

    def test_docker_stderr_logs_are_kept_as_evidence(self):
        completed = CompletedProcess(["docker", "logs"], 0, stdout="INFO ready\n", stderr="ERROR failed\n")
        with patch("hermes_sre_agent.live_backend.subprocess.run", return_value=completed):
            result = LiveReadOnlyBackend()._docker_logs("legal-agent", 20)
        self.assertEqual(result["lines"], ["INFO ready", "ERROR failed"])

    def test_stopped_container_and_running_resources_are_reported(self):
        backend = LiveReadOnlyBackend()

        def fake_run(command):
            if command[:3] == ["docker", "ps", "-a"]:
                return ('{"Names":"legal-agent","State":"exited","Status":"Exited (1)"}\n'
                        '{"Names":"reranker","State":"running","Status":"Up 2 minutes"}\n')
            return '{"Name":"reranker","CPUPerc":"70.2%","MemUsage":"2GiB / 4GiB","MemPerc":"50%"}\n'

        backend._run = fake_run
        result = backend._docker_list()
        self.assertEqual(result[0]["status"], "exited")
        self.assertEqual(result[1]["cpu"], "70.2%")
        self.assertEqual(result[1]["memory_percent"], "50%")

    def test_load_estimate_is_not_reported_as_cpu_usage(self):
        backend = LiveReadOnlyBackend()
        backend._linux_cpu_used_percent = lambda: None
        backend._memory_used_percent = lambda: 50.0
        with patch("hermes_sre_agent.live_backend.os.getloadavg", return_value=(8.0, 7.0, 6.0)), \
                patch("hermes_sre_agent.live_backend.os.cpu_count", return_value=8):
            result = backend._system_metrics()
        self.assertEqual(result["cpu_percent"], "未采集")
        self.assertEqual(result["load_based_estimate_percent"], 100.0)

    def test_config_diff_ignores_source_code_and_marks_git_as_unverified(self):
        backend = LiveReadOnlyBackend()
        backend._latest_git_diff = lambda: """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
-batch_size = 16
+batch_size = 64
diff --git a/config.yaml b/config.yaml
--- a/config.yaml
+++ b/config.yaml
-batch_size: 16
+batch_size: 64
"""
        result = backend._config_diff()
        self.assertEqual(result["changes"], [{"path": "config.yaml", "key": "batch_size", "before": 16, "after": 64}])
        self.assertFalse(result["runtime_config_verified"])

    def test_non_http_health_url_is_rejected(self):
        with self.assertRaisesRegex(ToolExecutionError, "仅支持 http 或 https"):
            LiveReadOnlyBackend()._service_health("file:///etc/passwd")

    def test_http_404_is_returned_as_service_evidence(self):
        error = HTTPError("http://localhost:8000/health", 404, "Not Found", None, None)
        with patch("hermes_sre_agent.live_backend._HEALTH_OPENER.open", side_effect=error):
            result = LiveReadOnlyBackend()._service_health("http://localhost:8000/health")

        self.assertEqual(result["status_code"], 404)
        self.assertIn("latency_ms", result)

    def test_code_review_diff_has_a_strict_size_limit(self):
        backend = LiveReadOnlyBackend()
        backend._run = lambda _command: "x" * 20_001

        result = backend._git_code_review()

        self.assertEqual(len(result["diff"]), 20_000)
        self.assertTrue(result["truncated"])
