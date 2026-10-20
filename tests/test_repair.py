"""修复生成、人工审批、冲突保护及真实沙箱闭环测试。"""
import json
import copy
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_sre_agent.repair import ChangeSet, PatchProposal, RepairWorkflow, replace_at, undo_applied_patch
from hermes_sre_agent.code_review import CodeReviewAgent
from hermes_sre_agent.web_server import ProjectRegistry, ReviewJobs


PASS = {"status": "finished", "exit_code": 0, "output": "1 passed", "coverage": {"main.py": [1]},
        "covered_changes": {"main.py": [1]}, "test_paths": ["tests/test_main.py"],
        "summary": "测试通过且执行到业务改动行。"}


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

    def test_undo_restores_modified_file_and_removes_created_file(self):
        proposal = PatchProposal({"main.py": (b"x = 1\n", b"x = 2\n"),
                                  "tests/test_new.py": (None, b"def test_new():\n    assert True\n")}, PASS)
        backup = proposal.apply(self.root, self.backups)
        result = undo_applied_patch(self.root, self.backups, proposal.id, proposal.public()["files"])
        self.assertEqual(result["restored_files"], 2)
        self.assertEqual((self.root / "main.py").read_bytes(), b"x = 1\n")
        self.assertFalse((self.root / "tests/test_new.py").exists())
        self.assertEqual(json.loads((Path(backup) / "manifest.json").read_text())["state"], "undone")
        with self.assertRaisesRegex(ValueError, "不匹配"):
            undo_applied_patch(self.root, self.backups, proposal.id, proposal.public()["files"])

    def test_undo_refuses_changed_file_before_restoring_any_file(self):
        proposal = PatchProposal({"main.py": (b"x = 1\n", b"x = 2\n"),
                                  "tests/test_new.py": (None, b"def test_new():\n    assert True\n")}, PASS)
        backup = proposal.apply(self.root, self.backups)
        (self.root / "main.py").write_text("user_change = True\n")
        with self.assertRaisesRegex(ValueError, "已有变化"):
            undo_applied_patch(self.root, self.backups, proposal.id, proposal.public()["files"])
        self.assertEqual((self.root / "main.py").read_text(), "user_change = True\n")
        self.assertTrue((self.root / "tests/test_new.py").exists())
        self.assertEqual(json.loads((Path(backup) / "manifest.json").read_text())["state"], "applied")

    def test_undo_failure_reinstates_files_already_removed(self):
        proposal = PatchProposal({"main.py": (b"x = 1\n", b"x = 2\n"),
                                  "tests/test_new.py": (None, b"def test_new():\n    assert True\n")}, PASS)
        backup = proposal.apply(self.root, self.backups)

        def fail_main_restore(fd, name, content, mode):
            if name == "main.py" and content == b"x = 1\n":
                raise OSError("模拟恢复失败")
            return replace_at(fd, name, content, mode)

        with patch("hermes_sre_agent.repair.replace_at", side_effect=fail_main_restore):
            with self.assertRaisesRegex(OSError, "模拟恢复失败"):
                undo_applied_patch(self.root, self.backups, proposal.id, proposal.public()["files"])
        self.assertEqual((self.root / "main.py").read_bytes(), b"x = 2\n")
        self.assertTrue((self.root / "tests/test_new.py").exists())
        self.assertEqual(json.loads((Path(backup) / "manifest.json").read_text())["state"], "applied")

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

    def test_invalid_indentation_is_retried_before_approval(self):
        source = ("def result(items):\n    for item in items:\n        if item == 'bad':\n"
                  "            return {'success': False}\n    return {'success': True}\n")
        (self.root / "main.py").write_text(source, encoding="utf-8")
        old = "            return {'success': False}\n    return {'success': True}\n"
        client = RecordingClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "patch", "changes": [{"path": "main.py", "old_text": old,
                                             "new_text": "    return {'success': False}\nreturn {'success': True}\n"}]},
            {"type": "patch", "changes": [{"path": "main.py", "old_text": old,
                                             "new_text": "            return {'success': False, 'error': 'bad'}\n"
                                                         "    return {'success': True, 'error': None}\n"}]},
        ])
        trace = []
        changes = ChangeSet(self.root)
        result = CodeReviewAgent(self.root, client, max_tool_calls=1, editor=changes,
                                 on_trace=trace.append).run("修正返回格式")
        self.assertIn("候选补丁", result["answer"])
        self.assertIn("'error': 'bad'", (self.root / "main.py").read_text(encoding="utf-8"))
        retries = [entry for entry in trace if entry["event"] == "patch_retry"]
        self.assertEqual([entry["details"]["error_code"] for entry in retries], ["invalid_python_syntax"])
        self.assertEqual(retries[0]["details"]["syntax_kind"], "IndentationError")
        self.assertIsInstance(retries[0]["details"]["syntax_line"], int)
        self.assertNotIn("return {'success'", json.dumps(retries, ensure_ascii=False))
        self.assertIn("IndentationError", client.prompts[-1][-1]["content"])
        self.assertIn("原前导空格", client.prompts[-1][-1]["content"])

    def test_missing_line_ending_gets_specific_retry_and_preserves_original(self):
        source = "def value():\n    return 1\n\nx = 2\n"
        (self.root / "main.py").write_text(source, encoding="utf-8")
        client = RecordingClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "patch", "changes": [{"path": "main.py", "start_line": 2, "end_line": 2,
                                             "new_text": "    return 2"}]},
            {"type": "patch", "changes": [{"path": "main.py", "start_line": 2, "end_line": 2,
                                             "new_text": "    return 2\n"}]},
        ])
        trace = []
        changes = ChangeSet(self.root)
        result = CodeReviewAgent(self.root, client, max_tool_calls=1, editor=changes,
                                 on_trace=trace.append).run("把返回值改为 2")
        self.assertIn("候选补丁", result["answer"])
        self.assertEqual((self.root / "main.py").read_text(encoding="utf-8"), source.replace("return 1", "return 2"))
        self.assertEqual(changes.changes()["main.py"][0], source.encode())
        self.assertEqual([entry["details"]["error_code"] for entry in trace if entry["event"] == "patch_retry"],
                         ["missing_line_ending"])
        self.assertIn("换行符结束", client.prompts[-1][-1]["content"])

    def test_blank_line_patch_is_blocked_before_testing_or_approval(self):
        (self.root / "main.py").write_text("def value():\n    return 1\n", encoding="utf-8")
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "edit_file", "arguments": {"path": "main.py", "old_text": "    return 1\n",
                                                              "new_text": "    return 1\n\n"}},
            {"type": "final", "answer": "候选修改已生成"},
        ])
        trace = []
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests") as tests:
            result, proposal = RepairWorkflow(self.root, client, lambda e: None, trace.append).run("修复", [])
        tests.assert_not_called()
        self.assertIsNone(proposal)
        self.assertEqual(result["outcome"], "blocked")
        self.assertIn("没有改变 Python 语法树", result["answer"])
        self.assertTrue(any(row["event"] == "cosmetic_patch_rejected" for row in trace))
        self.assertEqual((self.root / "main.py").read_text(encoding="utf-8"), "def value():\n    return 1\n")

    def test_unrelated_passing_test_cannot_authorize_writeback(self):
        (self.root / "test_harness.py").write_text("def test_unrelated():\n    assert 1 + 1 == 2\n")
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "edit_file", "arguments": {"path": "main.py", "old_text": "x = 1", "new_text": "x = 2"}},
            {"type": "final", "answer": "候选修改已生成"},
        ])
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests", return_value={
                "status": "finished", "exit_code": 0, "output": "1 passed", "coverage": {"main.py": []}}) as tests:
            _, proposal = RepairWorkflow(self.root, client, lambda e: None).run("修复 x", [])
        tests.assert_called_once_with(["test_harness.py", "tests/test_main.py"], coverage_targets=["main.py"])
        self.assertEqual(proposal.validation["failure_kind"], "change_not_covered")
        self.assertEqual(proposal.validation["exit_code"], 0)
        self.assertEqual(proposal.validation["defect_reproduction"], "not_run")
        self.assertFalse(proposal.public()["can_apply"])
        with self.assertRaises(ValueError):
            proposal.apply(self.root, self.backups)
        self.assertEqual((self.root / "main.py").read_text(), "x = 1\n")

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

    def test_non_python_edit_stops_before_tests_even_after_partial_python_patch(self):
        (self.root / "app.js").write_text("const value = 1;\n", encoding="utf-8")
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "edit_file", "arguments": {
                "path": "main.py", "old_text": "x = 1", "new_text": "x = 2"}},
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "app.js"}},
            {"type": "tool_call", "tool": "edit_file", "arguments": {
                "path": "app.js", "old_text": "value = 1", "new_text": "value = 2"}},
        ])
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests") as tests:
            result, proposal = RepairWorkflow(self.root, client, lambda e: None).run("修改 Python 和 JS", [])
        self.assertIsNone(proposal)
        self.assertEqual(result["error_kind"], "unsupported_edit_language")
        self.assertIn("本轮未完成全部目标", result["answer"])
        self.assertEqual(client.calls, 4)
        self.assertEqual((self.root / "main.py").read_text(), "x = 1\n")
        self.assertEqual((self.root / "app.js").read_text(), "const value = 1;\n")
        tests.assert_not_called()

    def test_patch_phase_unsupported_language_does_not_retry_or_approve(self):
        (self.root / "app.js").write_text("const value = 1;\n", encoding="utf-8")
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "app.js"}},
            {"type": "patch", "changes": [{"path": "app.js", "old_text": "value = 1", "new_text": "value = 2"}]},
        ])
        result = CodeReviewAgent(self.root, client, max_tool_calls=1,
                                 editor=ChangeSet(self.root)).run("修改 app.js")
        self.assertEqual(result["outcome"], "blocked")
        self.assertEqual(result["error_kind"], "unsupported_edit_language")
        self.assertEqual(client.calls, 2)
        self.assertEqual((self.root / "app.js").read_text(), "const value = 1;\n")

    def test_non_python_proposal_cannot_bypass_approval_scope(self):
        with self.assertRaisesRegex(ValueError, "仅支持修改 Python 文件"):
            PatchProposal({"app.js": (b"const value = 1;\n", b"const value = 2;\n")}, PASS)

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

    def test_auto_route_uses_verified_records_after_approval(self):
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
        self.assertIn("测试入口：`tests/test_main.py`", jobs.get(job_id)["answer"])
        self.assertIn("`main.py` 第 1 行", jobs.get(job_id)["answer"])
        self.assertIn("原缺陷是否曾被测试复现尚未证实", jobs.get(job_id)["answer"])
        self.assertNotIn("最终核对", jobs.get(job_id)["answer"])
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
