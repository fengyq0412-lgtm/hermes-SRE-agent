import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_sre_agent.code_review import CodeReviewAgent, SourceTools
from hermes_sre_agent.model_client import ModelRequestError
from hermes_sre_agent.repair import ChangeSet, RepairWorkflow


class ScriptedClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []
        self.calls = 0

    def complete(self, messages):
        self.messages = list(messages)
        self.calls += 1
        return json.dumps(next(self.replies), ensure_ascii=False)


class CodeReviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "main.py").write_text("def divide(x):\n    return 1 / x\n", encoding="utf-8")
        self.tools = SourceTools(self.root)

    def test_read_numbered_source_without_git_history(self):
        self.assertEqual(self.tools.list_files()["files"], ["main.py"])
        self.assertIn("2:     return 1 / x", self.tools.read_file("main.py")["content"])
        self.assertEqual(self.tools.search_code("return")["matches"][0]["line"], 2)

    def test_credentials_traversal_and_symlinks_are_not_read(self):
        (self.root / ".env").write_text("PRIVATE=do-not-read", encoding="utf-8")
        (self.root / "credentials.json").write_text("{}", encoding="utf-8")
        (self.root / "alias.py").symlink_to(self.root / "main.py")
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "dep.js").write_text("secret", encoding="utf-8")
        for name in (".env", "credentials.json", "../main.py", "alias.py", "node_modules/dep.js"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.tools.read_file(name)
        self.assertEqual(self.tools.list_files()["files"], ["main.py"])
        with self.assertRaises(ValueError):
            self.tools.call("run_shell", {"command": "echo forbidden"})

    def test_agent_reads_code_then_returns_model_answer_and_events(self):
        client = ScriptedClient([
            {"type": "tool_call", "tool": "list_files", "arguments": {}},
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "final", "answer": "[P2] main.py:2：x 为 0 时会抛出 ZeroDivisionError。"},
        ])
        events = []
        result = CodeReviewAgent(self.root, client, events.append).run("有哪些 bug？")
        self.assertIn("main.py:2", result["answer"])
        self.assertEqual(len(result["steps"]), 2)
        self.assertTrue(any("已读取 main.py" in e["message"] for e in events))
        self.assertIn("总体风险", result["answer"])

    def test_trace_records_decisions_without_raw_source_or_secret(self):
        (self.root / "main.py").write_text("PRIVATE_MARKER = 'sensitive-value'\n", encoding="utf-8")
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"},
             "reason": "先读取入口文件确定审查范围"},
            {"type": "final", "answer": "总体风险：待评估。", "reason": "已读取范围有限，保留残余风险"},
        ])
        trace = []
        CodeReviewAgent(self.root, client, on_trace=trace.append).run("检查")
        self.assertEqual([row["event"] for row in trace], ["review_start", "tool_result", "final"])
        self.assertEqual(trace[1]["details"]["reason"], "先读取入口文件确定审查范围")
        self.assertEqual(trace[1]["details"]["line_count"], 1)
        self.assertNotIn("sensitive-value", json.dumps(trace, ensure_ascii=False))
        self.assertNotIn("PRIVATE_MARKER", json.dumps(trace, ensure_ascii=False))
        self.assertEqual(CodeReviewAgent._safe_reason({"reason": "API_KEY=sk-secret"}),
                         "决策说明疑似包含敏感信息或代码，已省略")

    def test_outline_locates_python_functions_without_reading_entire_file(self):
        outline = self.tools.file_outline("main.py")
        self.assertEqual(outline["symbols"][0]["name"], "divide")
        self.assertEqual(outline["symbols"][0]["line"], 1)
        self.assertEqual(outline["symbols"][0]["end_line"], 2)

    def test_symbol_lookup_resolves_import_alias_to_real_function(self):
        (self.root / "main.py").write_text("from worker import actual as task_alias\n", encoding="utf-8")
        (self.root / "worker.py").write_text("def actual(value):\n    return value + 1\n", encoding="utf-8")
        result = self.tools.resolve_symbol("task_alias")
        self.assertFalse(result["truncated"])
        binding = result["matches"][0]
        self.assertEqual(binding["path"], "main.py")
        self.assertEqual(binding["imported_name"], "actual")
        self.assertEqual(binding["target_path"], "worker.py")
        self.assertEqual(binding["definition_line"], 1)

    def test_tool_budget_still_allows_a_final_synthesis_call(self):
        client = ScriptedClient([
            {"type": "tool_call", "tool": "list_files", "arguments": {}},
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "final", "answer": "总体风险：中。main.py:2 在 x=0 时抛出异常。"},
        ])
        events = []
        result = CodeReviewAgent(self.root, client, events.append, max_tool_calls=2).run("检查")
        self.assertEqual(len(result["steps"]), 2)
        self.assertIn("总体风险：中", result["answer"])
        self.assertTrue(any("评估风险" in event["message"] for event in events))

    def test_synthesis_retries_if_model_requests_another_tool(self):
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "search_code", "arguments": {"text": "divide"}},
            {"type": "final", "answer": "总体风险：待评估。已读取 main.py。"},
        ])
        result = CodeReviewAgent(self.root, client, max_tool_calls=1).run("检查")
        self.assertEqual(len(result["steps"]), 1)
        self.assertIn("待评估", result["answer"])

    def test_noncompliant_model_returns_scope_limited_report(self):
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
        ])
        result = CodeReviewAgent(self.root, client, max_tool_calls=1).run("检查")
        self.assertEqual(len(result["steps"]), 1)
        self.assertIn("待评估", result["answer"])
        self.assertIn("`main.py`", result["answer"])

    def test_invalid_arguments_do_not_crash_agent_and_can_be_corrected(self):
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py", "start_line": "a"}},
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "final", "answer": "main.py:2 除数可能为零。"},
        ])
        result = CodeReviewAgent(self.root, client).run("检查")
        self.assertIn("error", result["steps"][0])
        self.assertIn("main.py:2", result["answer"])

    def test_agent_cannot_finish_without_reading_any_source(self):
        client = ScriptedClient([{"type": "final", "answer": "代码没有 bug。"}] * 14)
        with self.assertRaises(ModelRequestError):
            CodeReviewAgent(self.root, client).run("检查")

    def test_malformed_tool_envelope_is_corrected_without_using_tool_budget(self):
        malformed = [
            {"type": "tool_call", "tool": "read_file", "arguments": None},
            {"type": "tool_call", "tool": "read_file", "arguments": '{"path":"main.py"}'},
            {"type": "tool_call", "tool": "read_file", "arguments": []},
            {"type": "tool_call", "arguments": {}},
            {"type": "tool_call", "tool": 42, "arguments": {}},
            {"type": "wrong", "tool": "read_file", "arguments": {}},
            {"type": "tool_call", "tool": "run_shell", "arguments": {"command": "敏感标记"}},
        ]
        for bad in malformed:
            with self.subTest(bad=bad):
                client = ScriptedClient([bad,
                    {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
                    {"type": "final", "answer": "已读取 main.py，除数为零时出错。"}])
                trace = []
                agent = CodeReviewAgent(self.root, client, max_tool_calls=1, on_trace=trace.append)
                with patch.object(agent.tools, "call", wraps=agent.tools.call) as execute:
                    result = agent.run("检查")
                execute.assert_called_once_with("read_file", {"path": "main.py"})
                self.assertEqual(len(result["steps"]), 1)
                self.assertEqual(client.calls, 3)
                self.assertEqual(sum(e["event"] == "protocol_retry" for e in trace), 1)
                self.assertNotIn("敏感标记", json.dumps(trace, ensure_ascii=False))
                self.assertTrue(any("本条请求未执行" in m["content"] for m in client.messages))

    def test_broken_json_can_be_corrected(self):
        replies = iter(['{"type":"tool_call",',
                        '{"type":"tool_call","tool":"read_file","arguments":{"path":"main.py"}}',
                        '{"type":"final","answer":"已读取代码。"}'])
        with patch.object(ScriptedClient, "complete", side_effect=lambda _: next(replies)):
            result = CodeReviewAgent(self.root, ScriptedClient([])).run("检查")
        self.assertEqual(len(result["steps"]), 1)

    def test_protocol_retries_exhaust_without_executing_or_dropping_evidence(self):
        bad = {"type": "tool_call", "tool": "read_file", "arguments": None}
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}}, bad, bad, bad])
        trace = []
        result = CodeReviewAgent(self.root, client, on_trace=trace.append).run("检查")
        self.assertEqual(result["outcome"], "blocked")
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(client.calls, 4)
        self.assertIn("main.py", result["answer"])
        self.assertEqual(sum(e["event"] == "protocol_retry" for e in trace), 2)
        self.assertEqual(sum(e["event"] == "protocol_retry_exhausted" for e in trace), 1)

    def test_protocol_retry_budget_does_not_reset_after_valid_tool(self):
        bad = {"type": "bad"}
        client = ScriptedClient([bad, {"type": "tool_call", "tool": "list_files", "arguments": {}},
                                bad, {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}}, bad])
        result = CodeReviewAgent(self.root, client).run("检查")
        self.assertEqual(result["outcome"], "blocked")
        self.assertEqual(client.calls, 5)

    def test_protocol_recovery_does_not_relax_edit_permissions(self):
        original = (self.root / "main.py").read_text()
        client = ScriptedClient([
            {"type": "tool_call", "tool": "edit_file", "arguments": None},
            {"type": "tool_call", "tool": "edit_file", "arguments": {
                "path": "main.py", "old_text": "return 1 / x", "new_text": "return 0"}},
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "final", "answer": "只读审查完成。"}])
        result = CodeReviewAgent(self.root, client).run("检查")
        self.assertIn("error", result["steps"][0])
        self.assertEqual((self.root / "main.py").read_text(), original)

    def test_existing_patch_survives_protocol_exhaustion_and_still_requires_validation(self):
        original = (self.root / "main.py").read_text()
        (self.root / "test_main.py").write_text("def test_placeholder():\n    assert True\n")
        bad = {"type": "tool_call", "tool": "read_file", "arguments": None}
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "edit_file", "arguments": {
                "path": "main.py", "old_text": "return 1 / x", "new_text": "return 0 if x == 0 else 1 / x"}}, bad, bad, bad])
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests", return_value={
                "status": "finished", "exit_code": 1, "output": "测试失败"}) as tests:
            result, proposal = RepairWorkflow(self.root, client, lambda e: None).run("修复", [])
        tests.assert_called_once()
        self.assertFalse(proposal.public()["can_apply"])
        self.assertEqual((self.root / "main.py").read_text(), original)
        self.assertEqual(sum(s["tool"] == "edit_file" for s in result["steps"]), 1)

    def test_provider_failure_is_not_mistaken_for_protocol_error(self):
        client = ScriptedClient([])
        with patch.object(client, "complete", side_effect=ModelRequestError("供应商失败")) as complete:
            with self.assertRaises(ModelRequestError):
                CodeReviewAgent(self.root, client).run("检查")
        complete.assert_called_once()

    def test_failed_edit_then_malformed_request_recovers_with_last_tool_slot(self):
        client = ScriptedClient([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "edit_file", "arguments": {
                "path": "main.py", "old_text": "不存在的旧代码", "new_text": "return 0"}},
            {"type": "tool_call", "tool": "edit_file", "arguments": []},
            {"type": "tool_call", "tool": "edit_file", "arguments": {
                "path": "main.py", "old_text": "return 1 / x", "new_text": "return 0 if x == 0 else 1 / x"}},
            {"type": "final", "answer": "临时副本已修改，尚需验证审批。"},
        ])
        changes = ChangeSet(self.root)
        result = CodeReviewAgent(self.root, client, editor=changes, max_tool_calls=3).run("修复")
        self.assertEqual(len(result["steps"]), 3)
        self.assertIn("error", result["steps"][1])
        self.assertIn("result", result["steps"][2])
        self.assertIn("main.py", changes.changes())

    def test_reading_has_line_limit(self):
        (self.root / "long.py").write_text("\n".join(["x = 1"] * 300), encoding="utf-8")
        result = self.tools.read_file("long.py", 1, 300)
        self.assertEqual(len(result["content"].splitlines()), 200)
        self.assertTrue(result["truncated"])
