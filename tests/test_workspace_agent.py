import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_sre_agent.workspace_agent import WorkspaceAgent, WorkspaceTools


class ScriptedClient:
    def __init__(self, replies):
        self.replies = iter(replies)

    def complete(self, messages):
        return json.dumps(next(self.replies), ensure_ascii=False)


class WorkspaceAgentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "main.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
        (self.root / ".env").write_text("KEY=private", encoding="utf-8")

    def test_task_uses_project_evidence_and_finishes(self):
        client = ScriptedClient([
            {"type": "tool_call", "tool": "list_files", "arguments": {}},
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "final", "answer": "main.py 定义 answer，返回 42。"},
        ])
        result = WorkspaceAgent(self.root, client).run("概括这个项目")
        self.assertEqual(result["outcome"], "completed")
        self.assertEqual([step["tool"] for step in result["steps"]], ["list_files", "read_file"])
        self.assertNotIn(".env", str(result["steps"]))

    def test_unknown_or_mutating_tools_never_execute(self):
        tools = WorkspaceTools(self.root)
        with self.assertRaises(ValueError):
            tools.call("run_shell", {"command": "touch pwned"})
        with self.assertRaises(ValueError):
            tools.call("read_file", {"path": ".env"})
        self.assertFalse((self.root / "pwned").exists())

    def test_command_tool_runs_only_in_sandbox_with_budget(self):
        tools = WorkspaceTools(self.root)
        with patch.object(tools.sandbox, "run_command", return_value={"status": "finished", "exit_code": 0,
                                                                         "output": "ok", "scope": "sandbox"}) as run:
            for _ in range(3):
                self.assertEqual(tools.call("run_command", {"argv": ["python", "-V"]})["exit_code"], 0)
            with self.assertRaisesRegex(ValueError, "3 次上限"):
                tools.call("run_command", {"argv": ["python", "-V"]})
            self.assertEqual(run.call_count, 3)

    def test_git_status_hides_excluded_paths_and_diff_is_scoped(self):
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        tools = WorkspaceTools(self.root)
        status = tools.call("git_status", {})
        self.assertEqual(status["files"], [{"status": "??", "path": "main.py"}])
        with self.assertRaises(ValueError):
            tools.call("git_diff", {"path": ".env"})

    def test_step_budget_forces_summary(self):
        client = ScriptedClient([
            {"type": "tool_call", "tool": "list_files", "arguments": {}},
            {"type": "final", "answer": "目前只列出了文件，尚未读取内容。"},
        ])
        result = WorkspaceAgent(self.root, client, max_steps=1).run("查看项目")
        self.assertEqual(result["outcome"], "completed")
        self.assertEqual(len(result["steps"]), 1)

    def test_web_research_records_opened_sources(self):
        url = "https://example.com/report"
        client = ScriptedClient([
            {"type": "tool_call", "tool": "search_web", "arguments": {"query": "latest report"}},
            {"type": "tool_call", "tool": "open_page", "arguments": {"url": url}},
            {"type": "final", "answer": "报告提出了一个结论。"},
        ])
        agent = WorkspaceAgent(self.root, client)
        def search(query):
            agent.tools.web.allowed_urls.add(url)
            return {"results": [{"title": "Report", "url": url}]}
        with patch.object(agent.tools.web, "search_web", side_effect=search), patch(
            "hermes_sre_agent.web_research._fetch_public_https", return_value=("<p>Evidence</p>", "text/html")
        ):
            result = agent.run("整理最新报告")
        self.assertEqual(result["outcome"], "completed")
        self.assertIn("[来源 1](https://example.com/report)", result["answer"])
