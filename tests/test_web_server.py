import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import time
from types import SimpleNamespace

from hermes_sre_agent.web_server import HermesRequestHandler, ProjectRegistry, ReviewJobs, browse_directories


class WebConsoleTests(unittest.TestCase):
    def test_local_api_rejects_other_origins_and_missing_session_token(self):
        handler = object.__new__(HermesRequestHandler)
        handler.server = SimpleNamespace(server_port=8765)
        responses = []
        handler._send_json = lambda status, body: responses.append(status)
        handler.headers = {"Host": "127.0.0.1:8765"}
        self.assertFalse(handler._valid_request(api=True))
        handler.headers["X-Hermes-Token"] = handler.session_token
        self.assertTrue(handler._valid_request(api=True))
        handler.headers["Origin"] = "https://untrusted.example"
        self.assertFalse(handler._valid_request(api=True))
        handler.headers = {"Host": "untrusted.example:8765", "X-Hermes-Token": handler.session_token}
        self.assertFalse(handler._valid_request(api=True))
        self.assertEqual(responses, [403, 403, 403])

    def test_project_selection_persists_without_env_configuration(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"HERMES_PROJECT_ROOTS": "/not/existing"}):
            storage = Path(directory) / "settings" / "projects.json"
            registry = ProjectRegistry(storage=storage)
            self.assertEqual(registry.list_public(), [])
            added = registry.add(directory)
            reloaded = ProjectRegistry(storage=storage)
            self.assertEqual(reloaded.resolve(added["id"]), Path(directory).resolve())

    def test_browser_only_lists_visible_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "project").mkdir()
            (root / ".private").mkdir()
            (root / "file.py").touch()
            result = browse_directories(directory)
            self.assertEqual([item["name"] for item in result["directories"]], ["project"])

    def test_followup_keeps_same_project_context_only(self):
        def fake_run(_self, question, history):
            return {"answer": "回答：" + question, "steps": []}

        with tempfile.TemporaryDirectory() as directory, patch("hermes_sre_agent.web_server.CodeReviewAgent.run", fake_run):
            root = Path(directory)
            (root / "second").mkdir()
            registry = ProjectRegistry()
            first = registry.add(directory)["id"]
            second = registry.add(str(root / "second"))["id"]
            jobs = ReviewJobs(registry, client_factory=lambda: object())
            try:
                initial = jobs.create(first, "找 bug")["id"]
                for _ in range(100):
                    if jobs.get(initial)["status"] == "completed":
                        break
                    time.sleep(.01)
                self.assertEqual(jobs.get(initial)["status"], "completed")
                followup = jobs.create(first, "如何触发？", initial)["id"]
                self.assertEqual(jobs.jobs[followup]["history"][0]["content"], "找 bug")
                with self.assertRaises(ValueError):
                    jobs.create(second, "不同项目的问题", initial)
            finally:
                jobs.executor.shutdown(wait=True)

    def test_registry_only_lists_configured_existing_projects(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            registry = ProjectRegistry(f"{first}{os.pathsep}/does-not-exist{os.pathsep}{second}")
            projects = registry.list_public()

        self.assertEqual(len(projects), 2)
        self.assertEqual({project["name"] for project in projects}, {Path(first).name, Path(second).name})
        with self.assertRaises(KeyError):
            registry.resolve("not-an-authorized-project")

    def test_console_asset_is_packaged_with_the_server(self):
        asset = HermesRequestHandler.asset_path
        content = asset.read_text(encoding="utf-8")
        self.assertIn("代码审查 Agent", content)
        self.assertIn("/api/reviews", content)
