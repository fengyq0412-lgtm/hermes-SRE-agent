import json
import tempfile
import unittest
from pathlib import Path

from hermes_sre_agent.code_review import CodeReviewAgent, SourceTools
from hermes_sre_agent.model_client import ModelRequestError


class ScriptedClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []

    def complete(self, messages):
        self.messages = list(messages)
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
        self.assertTrue(any("read_file 完成" in e["message"] for e in events))

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
        client = ScriptedClient([{"type": "final", "answer": "代码没有 bug。"}] * 12)
        with self.assertRaises(ModelRequestError):
            CodeReviewAgent(self.root, client).run("检查")

    def test_reading_has_line_limit(self):
        (self.root / "long.py").write_text("\n".join(["x = 1"] * 300), encoding="utf-8")
        result = self.tools.read_file("long.py", 1, 300)
        self.assertEqual(len(result["content"].splitlines()), 200)
        self.assertTrue(result["truncated"])
