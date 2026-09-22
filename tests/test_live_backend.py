import unittest
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

    def test_non_http_health_url_is_rejected(self):
        with self.assertRaisesRegex(ToolExecutionError, "仅支持 http 或 https"):
            LiveReadOnlyBackend()._service_health("file:///etc/passwd")

    def test_http_404_is_returned_as_service_evidence(self):
        error = HTTPError("http://localhost:8000/health", 404, "Not Found", None, None)
        with patch("hermes_sre_agent.live_backend.urlopen", side_effect=error):
            result = LiveReadOnlyBackend()._service_health("http://localhost:8000/health")

        self.assertEqual(result["status_code"], 404)
        self.assertIn("latency_ms", result)

    def test_code_review_diff_has_a_strict_size_limit(self):
        backend = LiveReadOnlyBackend()
        backend._run = lambda _command: "x" * 20_001

        result = backend._git_code_review()

        self.assertEqual(len(result["diff"]), 20_000)
        self.assertTrue(result["truncated"])
