"""修复生成、人工审批、冲突保护及真实沙箱闭环测试。"""
import json
import copy
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_sre_agent.repair import ChangeSet, PatchProposal, RepairWorkflow
from hermes_sre_agent.code_review import CodeReviewAgent
from hermes_sre_agent.web_server import ProjectRegistry, ReviewJobs


PASS = {"status": "finished", "exit_code": 0, "output": "1 passed"}


class ScriptedClient:
    """用确定性回复验证编排，避免测试依赖外部模型行为。"""
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = 0

    def complete(self, _messages):
        self.calls += 1
        return json.dumps(next(self.replies), ensure_ascii=False)


class RecordingClient(ScriptedClient):
    def __init__(self, replies):
        super().__init__(replies)
        self.prompts = []
        self.last_finish_reason = None

    def complete(self, messages):
        self.prompts.append(copy.deepcopy(messages))
        return super().complete(messages)


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "main.py").write_text("x = 1\n")
        (self.root / "tests").mkdir()
        (self.root / "tests" / "test_main.py").write_text("from main import x\ndef test_positive_value():\n    assert x > 0\n")
        self.backups = self.root / "backups"

    def proposal(self, validation=None):
        return PatchProposal({"main.py": (b"x = 1\n", b"x = 2\n")}, validation or PASS)

    def test_approval_applies_exact_diff_and_preserves_backup(self):
        proposal = self.proposal()
        backup = proposal.apply(self.root, self.backups)
        self.assertEqual((self.root / "main.py").read_text(), "x = 2\n")
        self.assertEqual((Path(backup) / "main.py").read_text(), "x = 1\n")
        with self.assertRaises(ValueError):
            proposal.apply(self.root, self.backups)

    def test_conflict_or_failed_tests_prevent_writeback(self):
        (self.root / "main.py").write_text("user_edit = True\n")
        with self.assertRaisesRegex(ValueError, "已变化"):
            self.proposal().apply(self.root, self.backups)
        self.assertEqual((self.root / "main.py").read_text(), "user_edit = True\n")
        with self.assertRaises(ValueError):
            self.proposal({"status": "finished", "exit_code": 1}).apply(self.root, self.backups)

    def test_symlink_edits_are_rejected_but_tests_are_editable(self):
        (self.root / "alias.py").symlink_to(self.root / "main.py")
        with self.assertRaises(OSError):
            PatchProposal({"alias.py": (b"x = 1\n", b"x = 2\n")}, PASS).apply(self.root, self.backups)
        changes = ChangeSet(self.root)
        changes.edit_file("tests/test_main.py", "assert x > 0", "assert x >= 1")
        self.assertIn("assert x >= 1", (self.root / "tests/test_main.py").read_text())
        with self.assertRaises(ValueError):
            changes.edit_file("../main.py", "x = 1", "x = 2")

    def test_advice_only_final_is_corrected_into_real_patch(self):
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "final", "answer": "建议把 x 改为 2。"},
            {"type": "tool_call", "tool": "edit_file", "arguments": {"path": "main.py", "old_text": "x = 1", "new_text": "x = 2"}},
            {"type": "final", "answer": "已在副本修改，等待审核。"},
        ])
        trace = []
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests", return_value=PASS) as tests:
            result, proposal = RepairWorkflow(self.root, client, lambda e: None, trace.append).run("修改 x 为 2", [])
        self.assertEqual(result["outcome"], "patch_ready")
        self.assertIn("+x = 2", proposal.public()["files"][0]["diff"])
        self.assertTrue(proposal.public()["can_apply"])
        self.assertEqual((self.root / "main.py").read_text(), "x = 1\n")
        self.assertEqual(sum(e["event"] == "missing_patch_retry" for e in trace), 1)
        tests.assert_called_once()

    def test_read_line_edit_reaches_sandbox_and_manual_approval(self):
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "edit_file", "arguments": {
                "path": "main.py", "start_line": 1, "end_line": 1, "new_text": "x = 2\n"}},
            {"type": "final", "answer": "已在临时副本修改，等待审核。"},
        ])
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests", return_value=PASS) as tests:
            result, proposal = RepairWorkflow(self.root, client, lambda e: None).run("把 x 改为 2", [])
        self.assertEqual(result["outcome"], "patch_ready")
        self.assertTrue(proposal.public()["can_apply"])
        self.assertIn("+x = 2", proposal.public()["files"][0]["diff"])
        self.assertEqual((self.root / "main.py").read_text(), "x = 1\n")
        tests.assert_called_once()

    def test_exhausted_research_budget_enters_patch_phase_and_rejects_more_reading(self):
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "search_code", "arguments": {"text": "x"}},
            {"type": "patch", "changes": [{"path": "main.py", "old_text": "x = 1", "new_text": "x = 2"}],
             "answer": "按要求将默认值改为2。"},
        ])
        trace = []
        changes = ChangeSet(self.root)
        result = CodeReviewAgent(self.root, client, max_tool_calls=1, editor=changes, on_trace=trace.append).run("把 x 改成2")
        self.assertEqual([s["tool"] for s in result["steps"]], ["read_file", "edit_file"])
        self.assertEqual((self.root / "main.py").read_text(), "x = 2\n")
        self.assertTrue(any(e["event"] == "patch_generation_started" for e in trace))
        self.assertTrue(any(e["event"] == "patch_retry" for e in trace))

    def test_patch_phase_uses_fresh_prompt_and_allows_two_focused_reads(self):
        (self.root / "worker.py").write_text("value = 1\n", encoding="utf-8")
        client = RecordingClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "worker.py"}},
            {"type": "patch", "changes": [{"path": "worker.py", "old_text": "value = 1", "new_text": "value = 2"}]},
        ])
        trace = []
        result = CodeReviewAgent(self.root, client, max_tool_calls=1, editor=ChangeSet(self.root),
                                 on_trace=trace.append).run("把 worker 的值改成2")
        self.assertEqual([step["tool"] for step in result["steps"]], ["read_file", "read_file", "edit_file"])
        self.assertIn("value = 2", (self.root / "worker.py").read_text())
        self.assertEqual(sum(e["event"] == "patch_retry" for e in trace), 0)
        self.assertNotIn("每轮回复 JSON", client.prompts[1][0]["content"])
        self.assertEqual(len(client.prompts[1]), 2)

    def test_patch_failures_log_safe_reason_and_return_specific_blocker(self):
        client = RecordingClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            *[{"type": "patch", "changes": [{"path": "main.py", "old_text": "missing",
                                                "new_text": "x = 2"}]}] * 3,
        ])
        trace = []
        result = CodeReviewAgent(self.root, client, max_tool_calls=1, editor=ChangeSet(self.root),
                                 on_trace=trace.append).run("修改")
        retries = [entry for entry in trace if entry["event"] == "patch_retry"]
        self.assertEqual([entry["details"]["error_code"] for entry in retries], ["old_text_not_unique"] * 3)
        self.assertNotIn("missing", json.dumps(retries, ensure_ascii=False))
        self.assertIn("精确且唯一", result["answer"])
        self.assertEqual((self.root / "main.py").read_text(), "x = 1\n")

    def test_old_text_failure_recovers_with_read_line_patch(self):
        client = RecordingClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "patch", "changes": [{"path": "main.py", "old_text": "1: x = 1",
                                            "new_text": "x = 2"}]},
            {"type": "patch", "changes": [{"path": "main.py", "start_line": 1,
                                            "end_line": 1, "new_text": "x = 2\n"}]},
        ])
        trace = []
        result = CodeReviewAgent(self.root, client, max_tool_calls=1, editor=ChangeSet(self.root),
                                 on_trace=trace.append).run("把 x 改成2")
        self.assertEqual((self.root / "main.py").read_text(), "x = 2\n")
        self.assertIn("已在临时副本", result["answer"])
        self.assertEqual([e["details"]["error_code"] for e in trace if e["event"] == "patch_retry"],
                         ["old_text_not_unique"])

    def test_line_patch_cannot_touch_unread_lines(self):
        (self.root / "main.py").write_text("x = 1\ny = 3\n", encoding="utf-8")
        changes = ChangeSet(self.root)
        changes.allow_range("main.py", 1, 1)
        with self.assertRaisesRegex(ValueError, "已读范围"):
            changes.apply_edits([{"path": "main.py", "start_line": 2, "end_line": 2,
                                  "new_text": "y = 4\n"}])
        self.assertEqual((self.root / "main.py").read_text(), "x = 1\ny = 3\n")

    def test_truncated_patch_is_reported_as_output_limit(self):
        class TruncatedClient(RecordingClient):
            def complete(self, messages):
                if self.calls:
                    self.prompts.append(copy.deepcopy(messages))
                    self.calls += 1
                    self.last_finish_reason = "length"
                    return '{"type":"patch","changes":['
                return super().complete(messages)

        client = TruncatedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
        ])
        trace = []
        result = CodeReviewAgent(self.root, client, max_tool_calls=1, editor=ChangeSet(self.root),
                                 on_trace=trace.append).run("修改")
        retries = [entry for entry in trace if entry["event"] == "patch_retry"]
        self.assertEqual(len(retries), 3)
        self.assertTrue(all(entry["details"]["error_code"] == "output_truncated" for entry in retries))
        self.assertIn("HERMES_MAX_OUTPUT_TOKENS", result["answer"])

    def test_invalid_batch_rolls_back_then_corrected_patch_applies(self):
        changes = ChangeSet(self.root)
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "patch", "changes": [
                {"path": "main.py", "old_text": "x = 1", "new_text": "x = 2"},
                {"path": "main.py", "old_text": "does_not_exist", "new_text": "bad"}]},
            {"type": "patch", "changes": [{"path": "main.py", "old_text": "x = 1", "new_text": "x = 3"}]},
        ])
        result = CodeReviewAgent(self.root, client, max_tool_calls=1, editor=changes).run("把 x 改成3")
        self.assertEqual((self.root / "main.py").read_text(), "x = 3\n")
        self.assertEqual(changes.changes()["main.py"][0], b"x = 1\n")
        self.assertEqual(len(result["steps"]), 2)

    def test_patch_phase_can_report_no_evidenced_fix_without_fabricating_changes(self):
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "blocked", "reason": "当前已读代码没有可确认的缺陷"},
        ])
        result = CodeReviewAgent(self.root, client, max_tool_calls=1, editor=ChangeSet(self.root)).run("找bug并修复")
        self.assertEqual(result["outcome"], "blocked")
        self.assertIn("没有可确认的缺陷", result["answer"])
        self.assertEqual((self.root / "main.py").read_text(), "x = 1\n")

    def test_false_missing_function_clarification_is_corrected_by_import_resolution(self):
        (self.root / "main.py").write_text("from worker import actual as task_alias\nresult = task_alias(1)\n", encoding="utf-8")
        (self.root / "worker.py").write_text("def actual(x):\n    return x - 1\n", encoding="utf-8")
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "clarification", "reason": "未找到 task_alias 的定义", "questions": ["task_alias 是否已实现？"]},
            {"type": "patch", "changes": [{"path": "worker.py", "old_text": "return x - 1", "new_text": "return x + 1"}],
             "answer": "已修复实际定义。"},
        ])
        trace = []
        result = CodeReviewAgent(self.root, client, max_tool_calls=1, editor=ChangeSet(self.root),
                                 on_trace=trace.append).run("修复 task_alias 的实现")
        self.assertIn("已修复实际定义", result["answer"])
        self.assertEqual((self.root / "worker.py").read_text(), "def actual(x):\n    return x + 1\n")
        self.assertTrue(any(e["event"] == "symbol_assumption_corrected" for e in trace))
        self.assertFalse(any(e["event"] == "clarification_requested" for e in trace))

    def test_false_missing_function_is_corrected_before_early_clarification(self):
        (self.root / "main.py").write_text("from worker import actual as task_alias\n", encoding="utf-8")
        (self.root / "worker.py").write_text("def actual(x):\n    return x - 1\n", encoding="utf-8")
        question = {"type": "clarification", "reason": "未找到 task_alias 的定义", "questions": ["task_alias 是否存在？"]}
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            question, question,
            {"type": "tool_call", "tool": "edit_file", "arguments": {"path": "worker.py", "old_text": "return x - 1", "new_text": "return x + 1"}},
            {"type": "final", "answer": "已修改目标定义。"},
        ])
        trace = []
        result = CodeReviewAgent(self.root, client, max_tool_calls=4, editor=ChangeSet(self.root),
                                 on_trace=trace.append).run("修复 task_alias")
        self.assertIn("已修改目标定义", result["answer"])
        self.assertTrue(any(e["event"] == "symbol_assumption_corrected" for e in trace))
        self.assertFalse(any(e["event"] == "clarification_requested" for e in trace))

    def test_noop_edit_and_repeated_completion_claims_are_blocked(self):
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "edit_file", "arguments": {"path": "main.py", "old_text": "x = 1", "new_text": "x = 1"}},
            *[{"type": "final", "answer": "已修复并完成。"}] * 3,
        ])
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests") as tests:
            result, proposal = RepairWorkflow(self.root, client, lambda e: None).run("修改", [])
        self.assertIsNone(proposal)
        self.assertEqual(result["outcome"], "blocked")
        self.assertNotIn("已修复并完成", result["answer"])
        self.assertEqual(client.calls, 5)
        tests.assert_not_called()

    def test_unsupported_edit_returns_specific_blocker(self):
        client = ScriptedClient([
            {"type": "tool_call", "tool": "list_files", "arguments": {}},
            {"type": "blocked", "reason": "当前不支持删除文件"},
            {"type": "blocked", "reason": "当前不支持删除文件"},
        ])
        result, proposal = RepairWorkflow(self.root, client, lambda e: None).run("删除一个文件", [])
        self.assertIsNone(proposal)
        self.assertEqual(result["outcome"], "blocked")
        self.assertIn("不支持删除文件", result["answer"])

    def test_missing_helper_is_reassessed_and_implemented_near_budget_limit(self):
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            *[{"type": "tool_call", "tool": "search_code", "arguments": {"text": "check_permission"}}] * 8,
            {"type": "blocked", "reason": "没有 check_permission 函数，无法加校验"},
            {"type": "tool_call", "tool": "edit_file", "arguments": {
                "path": "main.py", "old_text": "x = 1", "new_text":
                "def check_permission(user_id, owner_id):\n    return user_id is not None and user_id == owner_id\n\nx = 1"}},
            {"type": "final", "answer": "已依据用户给定规则在临时副本补充函数，待验证与审批。"},
        ])
        trace = []
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests", return_value=PASS):
            result, proposal = RepairWorkflow(self.root, client, lambda e: None, trace.append).run(
                "在 main.py 新增 check_permission 函数，只允许非空 user_id 等于 owner_id 时通过", [])
        self.assertEqual(result["outcome"], "patch_ready")
        self.assertIn("+def check_permission", proposal.public()["files"][0]["diff"])
        recovery = [e for e in trace if e["event"] == "blocker_reassessment"]
        self.assertEqual(len(recovery), 1)
        self.assertEqual(recovery[0]["details"]["remaining_calls"], 4)
        self.assertEqual((self.root / "main.py").read_text(), "x = 1\n")

    def test_missing_business_rule_asks_specific_question_after_history_check(self):
        question = "删除权限是仅创建人可用，还是同一团队成员都可用？"
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "blocked", "reason": "没有找到权限规则"},
            {"type": "tool_call", "tool": "search_history", "arguments": {"text": "权限"}},
            {"type": "clarification", "reason": "源码与历史都没有定义资源操作权限", "questions": [question]},
        ])
        searches = []
        def search(text):
            searches.append(text)
            return {"matches": [], "limit_reached": False}
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests") as tests:
            result, proposal = RepairWorkflow(self.root, client, lambda e: None, history_search=search).run("加上权限校验", [])
        self.assertEqual(result["outcome"], "needs_input")
        self.assertEqual(result["questions"], [question])
        self.assertEqual(searches, ["权限"])
        self.assertIsNone(proposal)
        tests.assert_not_called()

    def test_clarification_can_be_answered_in_same_conversation(self):
        registry = ProjectRegistry()
        project = registry.add(str(self.root))
        jobs = ReviewJobs(registry, client_factory=lambda: object())
        self.addCleanup(jobs.executor.shutdown)
        result = {"outcome": "needs_input", "answer": "用户身份从哪里取得？", "questions": ["用户身份从哪里取得？"], "steps": []}
        with patch("hermes_sre_agent.web_server.RepairWorkflow.run", return_value=(result, None)):
            job_id = jobs.create(project["id"], "加权限校验", mode="repair")["id"]
            for _ in range(100):
                if jobs.get(job_id)["status"] not in {"queued", "running"}:
                    break
                time.sleep(.01)
        self.assertEqual(jobs.get(job_id)["status"], "needs_input")
        with patch.object(jobs.executor, "submit"):
            followup = jobs.create(project["id"], "从已验证的会话取得当前用户", parent_id=job_id)["id"]
        self.assertIn("用户身份从哪里取得", jobs.jobs[followup]["history"][-1]["content"])
        self.assertEqual(jobs.jobs[followup]["conversation_id"], jobs.jobs[job_id]["conversation_id"])

    def test_blocked_modification_is_visible_and_accepts_followup(self):
        registry = ProjectRegistry()
        project = registry.add(str(self.root))
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            *[{"type": "final", "answer": "建议改成 2"}] * 3,
        ])
        jobs = ReviewJobs(registry, client_factory=lambda: client)
        self.addCleanup(jobs.executor.shutdown)
        job_id = jobs.create(project["id"], "修改 x 为 2", mode="repair")["id"]
        for _ in range(100):
            job = jobs.get(job_id)
            if job["status"] not in {"queued", "running"}:
                break
            time.sleep(.01)
        self.assertEqual(job["status"], "blocked")
        self.assertIsNone(job["proposal"])
        self.assertIn("修改任务未完成", job["answer"])
        with patch.object(jobs.executor, "submit"):
            followup = jobs.create(project["id"], "那先解释一下", parent_id=job_id)["id"]
        self.assertIn("修改任务未完成", jobs.jobs[followup]["history"][-1]["content"])

    def test_rejected_or_wrong_proposal_cannot_be_applied(self):
        registry = ProjectRegistry()
        project = registry.add(str(self.root))
        jobs = ReviewJobs(registry)
        self.addCleanup(jobs.executor.shutdown)
        proposal = self.proposal()
        jobs.proposals["job"] = proposal
        jobs.jobs["job"] = {"project_id": project["id"], "status": "awaiting_review", "events": []}
        with self.assertRaises(ValueError):
            jobs.decide("job", "forged-id", "approve")
        jobs.decide("job", proposal.id, "reject")
        with self.assertRaises(ValueError):
            jobs.decide("job", proposal.id, "approve")
        self.assertEqual((self.root / "main.py").read_text(), "x = 1\n")

    def test_background_job_waits_for_human_decision(self):
        registry = ProjectRegistry()
        project = registry.add(str(self.root))
        jobs = ReviewJobs(registry, client_factory=lambda: object())
        self.addCleanup(jobs.executor.shutdown)
        proposal = self.proposal()
        with patch("hermes_sre_agent.web_server.RepairWorkflow.run", return_value=({"answer": "待审核", "steps": []}, proposal)):
            job_id = jobs.create(project["id"], "修复", mode="repair")["id"]
            for _ in range(100):
                if jobs.get(job_id)["status"] not in {"queued", "running"}:
                    break
                time.sleep(.01)
        self.assertEqual(jobs.get(job_id)["status"], "awaiting_review")
        self.assertEqual((self.root / "main.py").read_text(), "x = 1\n")
        with patch("hermes_sre_agent.web_server.Path.cwd", return_value=self.root):
            decided = jobs.decide(job_id, proposal.id, "approve")
        self.assertIn(decided["status"], {"running", "completed"})
        for _ in range(100):
            if jobs.get(job_id)["status"] == "completed":
                break
            time.sleep(.01)
        self.assertEqual(jobs.get(job_id)["status"], "completed")
        self.assertEqual((self.root / "main.py").read_text(), "x = 2\n")

    def test_auto_route_pauses_then_resumes_model_after_approval(self):
        registry = ProjectRegistry()
        project = registry.add(str(self.root))
        replies = iter([
            json.dumps({"mode": "repair"}),
            json.dumps({"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}}),
            json.dumps({"type": "final", "answer": "最终核对：main.py 已写回 x = 2。"}),
        ])
        class Client:
            def complete(self, _messages):
                return next(replies)
        jobs = ReviewJobs(registry, client_factory=Client)
        self.addCleanup(jobs.executor.shutdown)
        proposal = self.proposal()
        with patch("hermes_sre_agent.web_server.RepairWorkflow.run", return_value=({"answer": "候选修复等待审核", "steps": []}, proposal)):
            job_id = jobs.create(project["id"], "修复 main.py", mode="auto")["id"]
            for _ in range(100):
                if jobs.get(job_id)["status"] == "awaiting_review":
                    break
                time.sleep(.01)
        self.assertEqual(jobs.get(job_id)["mode"], "repair")
        self.assertEqual((self.root / "main.py").read_text(), "x = 1\n")
        with patch("hermes_sre_agent.web_server.Path.cwd", return_value=self.root):
            jobs.decide(job_id, proposal.id, "approve")
        for _ in range(100):
            if jobs.get(job_id)["status"] == "completed":
                break
            time.sleep(.01)
        self.assertIn("最终核对", jobs.get(job_id)["answer"])
        self.assertEqual((self.root / "main.py").read_text(), "x = 2\n")

    def test_multi_file_failure_rolls_back_completed_writes(self):
        from hermes_sre_agent.repair import replace_at
        (self.root / "second.py").write_text("y = 1\n")
        proposal = PatchProposal({"main.py": (b"x = 1\n", b"x = 2\n"),
                                  "second.py": (b"y = 1\n", b"y = 2\n")}, PASS)
        def fail_second(fd, name, content, mode):
            if name == "second.py":
                raise OSError("模拟磁盘写入失败")
            return replace_at(fd, name, content, mode)
        with patch("hermes_sre_agent.repair.replace_at", side_effect=fail_second):
            with self.assertRaises(OSError):
                proposal.apply(self.root, self.backups)
        self.assertEqual((self.root / "main.py").read_text(), "x = 1\n")

    @unittest.skipUnless(os.environ.get("HERMES_SANDBOX_INTEGRATION") == "1", "需要真实 Docker")
    def test_model_edit_sandbox_verify_then_human_apply(self):
        (self.root / "main.py").write_text("def increment(x):\n    return x - 1\n")
        (self.root / "tests").mkdir(exist_ok=True)
        (self.root / "tests" / "test_main.py").write_text("from main import increment\ndef test_increment():\n    assert increment(1) == 2\n")
        replies = iter([
            # 重现模型把取证预算全部用于读取的情况，确认独立补丁阶段仍可完成闭环。
            *[{"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}}] * 10,
            {"type": "patch", "changes": [{"path": "main.py", "old_text": "return x - 1", "new_text": "return x + 1"}],
             "answer": "已在临时副本修正递增逻辑，等待验证与人工审核。"},
        ])
        class Client:
            def complete(self, _messages):
                return json.dumps(next(replies))
        result, proposal = RepairWorkflow(self.root, Client(), lambda e: None).run("修复递增函数", [])
        self.assertEqual(proposal.validation["exit_code"], 0, proposal.validation)
        self.assertIn("return x - 1", (self.root / "main.py").read_text())
        self.assertIn("+    return x + 1", proposal.public()["files"][0]["diff"])
        proposal.apply(self.root, self.backups)
        self.assertIn("return x + 1", (self.root / "main.py").read_text())
