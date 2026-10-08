import errno
import io
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch
import time
from types import SimpleNamespace

from hermes_sre_agent.web_server import (HermesRequestHandler, ProjectRegistry, ReviewJobs,
                                         SERVER_LOGGER, bind_web_server, browse_directories,
                                         configure_server_logging, main, restartable_listener)


class WebConsoleTests(unittest.TestCase):
    def test_backend_log_is_private_and_excludes_request_content(self):
        old_handlers = list(SERVER_LOGGER.handlers)
        old_level, old_propagate = SERVER_LOGGER.level, SERVER_LOGGER.propagate
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = configure_server_logging(Path(directory) / ".hermes" / "server.log", also_console=False)
                SERVER_LOGGER.info("event=api_response method=POST route=/api/reviews status=202 bytes=42")
                for handler in SERVER_LOGGER.handlers:
                    handler.flush()
                content = path.read_text(encoding="utf-8")
                mode = path.stat().st_mode & 0o777
        finally:
            for handler in SERVER_LOGGER.handlers:
                handler.close()
                owned_stream = getattr(handler, "_hermes_owned_stream", None)
                if owned_stream is not None:
                    owned_stream.close()
            SERVER_LOGGER.handlers.clear()
            SERVER_LOGGER.setLevel(old_level)
            SERVER_LOGGER.propagate = old_propagate
            for handler in old_handlers:
                SERVER_LOGGER.addHandler(handler)

        self.assertIn("event=api_response", content)
        self.assertIn("route=/api/reviews", content)
        self.assertNotIn("用户问题", content)
        self.assertEqual(mode, 0o600)

    def test_routing_distinguishes_advice_from_explicit_execution(self):
        cases = [
            ("请帮我修复 main.py 的边界错误", '{"mode":"review"}', "repair"),
            ("只给修复建议，不要改代码", '{"mode":"repair"}', "review"),
            ("如何修复这个 bug？", "无法识别", "review"),
            ("只分析这个项目", "无法识别", "review"),
            ("修复 main.py", "无法识别", "repair"),
            ("检查一下有哪些 bug", '{"mode":"review"}', "review"),
            ("就按你上面的方案改", '{"mode":"repair"}', "repair"),
        ]
        for question, reply, expected in cases:
            with self.subTest(question=question):
                client = SimpleNamespace(complete=lambda messages: reply)
                mode, _, _ = ReviewJobs._choose_mode(client, question, [])
                self.assertEqual(mode, expected)

    def test_port_conflict_shows_actionable_message(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"HERMES_WEB_PORT": "8765"}), \
                patch("hermes_sre_agent.web_server.load_dotenv"), \
                patch("hermes_sre_agent.web_server.ThreadingHTTPServer",
                      side_effect=OSError(errno.EADDRINUSE, "Address already in use")), \
                patch("hermes_sre_agent.web_server.restartable_listener", return_value=None), \
                patch("hermes_sre_agent.web_server.os.kill") as stop, \
                patch("hermes_sre_agent.web_server.Path.cwd", return_value=Path(directory)):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertIn("端口 8765 已被占用", str(error.exception))
        self.assertIn("HERMES_WEB_PORT", str(error.exception))
        stop.assert_not_called()

    def test_existing_hermes_is_stopped_before_rebinding(self):
        conflict = OSError(errno.EADDRINUSE, "Address already in use")
        server = object()
        with patch("hermes_sre_agent.web_server.ThreadingHTTPServer", side_effect=[conflict, server]), \
                patch("hermes_sre_agent.web_server.restartable_listener", return_value=1234), \
                patch("hermes_sre_agent.web_server.os.kill") as stop:
            self.assertIs(bind_web_server(8765), server)
        stop.assert_called_once()
        self.assertEqual(stop.call_args.args[0], 1234)

    def test_listener_must_match_same_user_and_entry_script(self):
        with patch("hermes_sre_agent.web_server.sys.argv", ["/tmp/hermes-web"]), \
                patch("hermes_sre_agent.web_server.os.getuid", return_value=501), \
                patch("hermes_sre_agent.web_server.os.getpid", return_value=999), \
                patch("hermes_sre_agent.web_server.subprocess.run", side_effect=[
                    SimpleNamespace(returncode=0, stdout="1234\n"),
                    SimpleNamespace(returncode=0, stdout="501 /usr/bin/python3 /tmp/hermes-web\n"),
                ]):
            self.assertEqual(restartable_listener(8765), 1234)
        with patch("hermes_sre_agent.web_server.sys.argv", ["/tmp/hermes-web"]), \
                patch("hermes_sre_agent.web_server.os.getuid", return_value=501), \
                patch("hermes_sre_agent.web_server.subprocess.run", side_effect=[
                    SimpleNamespace(returncode=0, stdout="1234\n"),
                    SimpleNamespace(returncode=0, stdout="501 /usr/bin/python3 /tmp/other-web\n"),
                ]):
            self.assertIsNone(restartable_listener(8765))

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

    def test_cancel_running_job_keeps_interrupted_result_and_allows_followup(self):
        started, release = Event(), Event()

        class SlowClient:
            last_usage = None

            def complete(self, _messages):
                started.set()
                release.wait(5)
                return '{"type":"final","answer":"迟到的回答"}'

        with tempfile.TemporaryDirectory() as directory:
            registry = ProjectRegistry()
            project_id = registry.add(directory)["id"]
            jobs = ReviewJobs(registry, client_factory=SlowClient,
                              conversation_path=Path(directory) / "conversations.sqlite3")
            try:
                created = jobs.create(project_id, "检查代码", mode="review")
                self.assertTrue(started.wait(2), "模型调用未启动")
                stopped = jobs.cancel(project_id, created["id"])
                self.assertEqual(stopped["status"], "interrupted")
                self.assertIn("可在当前会话继续", stopped["answer"])
                with self.assertRaisesRegex(ValueError, "暂不能删除"):
                    jobs.delete_conversation(project_id, created["conversation_id"])
                release.set()
                jobs.futures[created["id"]].result(timeout=2)
                self.assertEqual(jobs.get(created["id"])["status"], "interrupted")
                self.assertNotIn("迟到的回答", jobs.get(created["id"])["answer"])
                with patch.object(jobs.executor, "submit"):
                    continued = jobs.create(project_id, "继续检查", parent_id=created["id"], mode="review")
                self.assertEqual(continued["conversation_id"], created["conversation_id"])
                self.assertEqual(jobs.jobs[continued["id"]]["history"][-1]["content"].splitlines()[0],
                                 "[本轮实际状态：interrupted]")
            finally:
                release.set()
                jobs.executor.shutdown(wait=True)

    def test_cancel_route_requires_project_and_returns_updated_job(self):
        handler = object.__new__(HermesRequestHandler)
        handler.server = SimpleNamespace(server_port=8765)
        handler.path = "/api/reviews/job-1/cancel"
        payload = b'{"project_id":"project-1"}'
        handler.headers = {"Host": "127.0.0.1:8765", "X-Hermes-Token": handler.session_token,
                           "Content-Type": "application/json", "Content-Length": str(len(payload))}
        handler.rfile = io.BytesIO(payload)
        handler.jobs = SimpleNamespace(cancel=lambda project_id, job_id: {
            "project_id": project_id, "id": job_id, "status": "interrupted"})
        responses = []
        handler._send_json = lambda status, body: responses.append((status, body))
        handler.do_POST()
        self.assertEqual(responses, [(200, {"project_id": "project-1", "id": "job-1",
                                           "status": "interrupted"})])

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

    def test_completed_conversation_can_be_deleted_with_its_trace_and_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ProjectRegistry()
            project_id = registry.add(directory)["id"]
            traces = root / ".hermes" / "traces"
            jobs = ReviewJobs(registry, trace_dir=traces,
                              conversation_path=root / ".hermes" / "conversations.sqlite3")
            job = {"id": "a" * 32, "project_id": project_id, "conversation_id": "b" * 32,
                   "created_at": time.time(), "question": "已经完成的问题", "status": "completed",
                   "answer": "已完成", "events": [], "trace": [], "steps": [], "usage": {},
                   "mode": "review", "proposal": None}
            try:
                jobs.jobs[job["id"]] = job
                jobs.store.save(job)
                traces.mkdir(parents=True)
                trace_path = traces / f"{job['id']}.jsonl"
                trace_path.write_text('{"event":"completed"}\n', encoding="utf-8")

                deleted = jobs.delete_conversation(project_id, job["conversation_id"])

                self.assertEqual(deleted, {"id": job["conversation_id"], "deleted_turns": 1})
                self.assertNotIn(job["id"], jobs.jobs)
                self.assertEqual(jobs.store.load(), [])
                self.assertFalse(trace_path.exists())
            finally:
                jobs.executor.shutdown(wait=True)

    def test_active_conversation_cannot_be_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ProjectRegistry()
            project_id = registry.add(directory)["id"]
            jobs = ReviewJobs(registry, client_factory=lambda: object())
            job = {"id": "c" * 32, "project_id": project_id, "conversation_id": "d" * 32,
                   "created_at": time.time(), "question": "仍在运行的问题", "status": "running"}
            try:
                jobs.jobs[job["id"]] = job
                with self.assertRaisesRegex(ValueError, "暂不能删除"):
                    jobs.delete_conversation(project_id, job["conversation_id"])
                self.assertIn(job["id"], jobs.jobs)
            finally:
                jobs.executor.shutdown(wait=True)

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

    def test_review_usage_counts_all_model_calls_and_unknown_responses(self):
        class FakeClient:
            def __init__(self):
                self.last_usage = None
                self.calls = 0

            def complete(self, _messages):
                self.calls += 1
                self.last_usage = ({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
                                   if self.calls == 1 else None)
                return "回答"

        def fake_run(agent, _question, _history):
            agent.client.complete([])
            agent.client.complete([])
            return {"answer": "报告", "steps": []}

        with tempfile.TemporaryDirectory() as directory, patch("hermes_sre_agent.web_server.CodeReviewAgent.run", fake_run):
            registry = ProjectRegistry()
            project_id = registry.add(directory)["id"]
            jobs = ReviewJobs(registry, client_factory=FakeClient)
            try:
                job_id = jobs.create(project_id, "审查代码")["id"]
                for _ in range(100):
                    job = jobs.get(job_id)
                    if job["status"] == "completed":
                        break
                    time.sleep(.01)
                self.assertEqual(job["status"], "completed")
                self.assertEqual(job["usage"]["requests"], 2)
                self.assertEqual(job["usage"]["reported_requests"], 1)
                self.assertEqual(job["usage"]["unreported_requests"], 1)
                self.assertEqual(job["usage"]["total_tokens"], 120)
                self.assertEqual(job["session_usage"], jobs.get_usage())
            finally:
                jobs.executor.shutdown(wait=True)

    def test_trace_is_persisted_as_ordered_jsonl_without_model_payload(self):
        class FakeClient:
            def __init__(self):
                self.last_usage = None
                self.calls = 0

            def complete(self, _messages):
                self.calls += 1
                self.last_usage = {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}
                answers = [
                    {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"},
                     "reason": "读取主入口验证代码路径"},
                    {"type": "final", "answer": "总体风险：待评估。", "reason": "证据范围有限"},
                ]
                return json.dumps(answers[self.calls - 1], ensure_ascii=False)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").write_text("SECRET_VALUE = 'do-not-log'\n", encoding="utf-8")
            registry = ProjectRegistry()
            project_id = registry.add(directory)["id"]
            jobs = ReviewJobs(registry, client_factory=FakeClient, trace_dir=root / ".hermes" / "traces")
            try:
                job_id = jobs.create(project_id, "审查代码")['id']
                for _ in range(100):
                    job = jobs.get(job_id)
                    if job["status"] == "completed":
                        break
                    time.sleep(.01)
                self.assertEqual(job["status"], "completed")
                rows = [json.loads(line) for line in Path(job["trace_path"]).read_text(encoding="utf-8").splitlines()]
                self.assertEqual(rows, job["trace"])
                self.assertEqual([row["seq"] for row in rows], list(range(1, len(rows) + 1)))
                self.assertEqual(len([row for row in rows if row["event"] == "usage"]), 2)
                self.assertNotIn("do-not-log", json.dumps(rows, ensure_ascii=False))
                self.assertNotIn("SECRET_VALUE", json.dumps(rows, ensure_ascii=False))
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
        self.assertIn("项目 Agent", content)
        self.assertIn("需要你批准代码修改", content)
        self.assertNotIn("id=\"task-mode\"", content)
        self.assertIn("/api/reviews", content)
        self.assertIn("本次审查", content)
