"""验证新建、测试修改、冲突与回滚不会覆盖或删除范围外文件。"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_sre_agent.code_review import CodeReviewAgent
from hermes_sre_agent.repair import ChangeSet, PatchProposal, RepairWorkflow, create_at
from hermes_sre_agent.sandbox import DockerSandbox


PASS = {"status": "finished", "exit_code": 0, "output": "1 passed"}


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)

    def complete(self, messages):
        return json.dumps(next(self.replies), ensure_ascii=False)


class FilePermissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "main.py").write_text("x = 1\n")
        self.backups = self.root / ".hermes/backups"

    def test_new_file_only_appears_in_original_after_approval(self):
        proposal = PatchProposal({"tests/test_main.py": (None, b"def test_value():\n    assert 1 + 1 == 2\n")}, PASS)
        public = proposal.public()
        self.assertEqual(public["files"][0]["operation"], "create")
        self.assertIn("--- /dev/null", public["files"][0]["diff"])
        self.assertFalse((self.root / "tests").exists())
        backup = proposal.apply(self.root, self.backups)
        self.assertTrue((self.root / "tests/test_main.py").exists())
        manifest = json.loads((Path(backup) / "manifest.json").read_text())
        self.assertEqual(manifest["operations"]["tests/test_main.py"], "create")
        self.assertEqual((self.root / "main.py").read_text(), "x = 1\n")

    def test_new_file_collision_preserves_user_file(self):
        proposal = PatchProposal({"new.py": (None, b"x = 2\n")}, PASS)
        (self.root / "new.py").write_text("user_change = True\n")
        with self.assertRaises(ValueError):
            proposal.apply(self.root, self.backups)
        self.assertEqual((self.root / "new.py").read_text(), "user_change = True\n")

    def test_collision_after_preflight_does_not_overwrite_target(self):
        proposal = PatchProposal({"new.py": (None, b"x = 2\n")}, PASS)
        def racing_create(fd, name, content, mode):
            (self.root / "new.py").write_text("user_change = True\n")
            return create_at(fd, name, content, mode)
        with patch("hermes_sre_agent.repair.create_at", side_effect=racing_create):
            with self.assertRaises(FileExistsError):
                proposal.apply(self.root, self.backups)
        self.assertEqual((self.root / "new.py").read_text(), "user_change = True\n")

    def test_unsafe_paths_and_symlink_parent_are_rejected(self):
        changes = ChangeSet(self.root)
        for path in ("../outside.py", "/tmp/outside.py", ".git/hook.py", ".env.py", "nested/../main.py"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                changes.create_file(path, "x = 2\n")
        with tempfile.TemporaryDirectory() as outside:
            (self.root / "tests").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(OSError):
                changes.create_file("tests/test_new.py", "x = 2\n")
            with self.assertRaises(OSError):
                PatchProposal({"tests/test_new.py": (None, b"x = 2\n")}, PASS).apply(self.root, self.backups)
            self.assertEqual(list(Path(outside).iterdir()), [])

    def test_batch_failure_removes_only_new_temporary_files(self):
        changes = ChangeSet(self.root)
        with self.assertRaises(ValueError):
            changes.apply_edits([{"path": "tests/test_new.py", "content": "x = 1\n"},
                                 {"path": "main.py", "old_text": "missing", "new_text": "x = 2"}])
        self.assertFalse((self.root / "tests/test_new.py").exists())
        with self.assertRaises(FileExistsError):
            changes.apply_edits([{"path": "main.py", "content": "overwrite = True\n"}])
        self.assertEqual((self.root / "main.py").read_text(), "x = 1\n")
        self.assertEqual(changes.changes(), {})

    def test_apply_failure_rolls_back_creation_without_deleting_other_files(self):
        (self.root / "keep.txt").write_text("user data")
        proposal = PatchProposal({"tests/test_new.py": (None, b"x = 2\n"),
                                  "main.py": (b"x = 1\n", b"x = 2\n")}, PASS)
        with patch("hermes_sre_agent.repair.replace_at", side_effect=OSError("模拟写入失败")):
            with self.assertRaises(OSError):
                proposal.apply(self.root, self.backups)
        self.assertFalse((self.root / "tests").exists())
        self.assertEqual((self.root / "keep.txt").read_text(), "user data")
        self.assertEqual((self.root / "main.py").read_text(), "x = 1\n")

    def test_test_generation_scope_cannot_change_business_or_empty_files(self):
        changes = ChangeSet(self.root)
        changes.test_only = True
        with self.assertRaises(ValueError):
            changes.edit_file("main.py", "x = 1", "x = 2")
        with self.assertRaises(ValueError):
            changes.create_file("other.py", "x = 2\n")
        changes.create_file("tests/test_new.py", "def test_value():\n    assert 2 > 1\n")
        with self.assertRaises(ValueError):
            changes.edit_file("tests/test_new.py", (self.root / "tests/test_new.py").read_text(), "")

    def test_stale_line_ranges_require_rereading(self):
        changes = ChangeSet(self.root)
        changes.allow_range("main.py", 1, 1)
        changes.edit_file("main.py", "x = 1", "x = 2")
        with self.assertRaisesRegex(ValueError, "当前版本"):
            changes.edit_lines("main.py", 1, 1, "x = 3\n")

    def test_patch_batch_can_create_tests(self):
        changes = ChangeSet(self.root)
        client = Client([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "patch", "changes": [
                {"path": "main.py", "old_text": "x = 1", "new_text": "x = 2"},
                {"path": "tests/test_main.py", "content": "from main import x\ndef test_value():\n    assert x == 2\n"}]},
        ])
        CodeReviewAgent(self.root, client, editor=changes, max_tool_calls=1).run("修改并补充测试")
        self.assertIsNone(changes.changes()["tests/test_main.py"][0])

    def test_missing_tests_are_generated_and_reviewed_with_business_patch(self):
        client = Client([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "edit_file", "arguments": {"path": "main.py", "old_text": "x = 1", "new_text": "x = 2"}},
            {"type": "final", "answer": "已在副本修改默认值"},
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "create_file", "arguments": {"path": "tests/test_main.py", "content":
                "from main import x\ndef test_value():\n    assert x == 2\n"}},
            {"type": "final", "answer": "已补充回归测试"},
        ])
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests", return_value=dict(PASS)) as run:
            result, proposal = RepairWorkflow(self.root, client, lambda e: None).run("把 x 改成 2", [])
        self.assertEqual(result["outcome"], "patch_ready")
        self.assertEqual(len(proposal.public()["files"]), 2)
        run.assert_called_once_with(["tests/test_main.py"])
        self.assertFalse((self.root / "tests").exists())
        proposal.apply(self.root, self.backups)
        self.assertTrue((self.root / "tests/test_main.py").exists())

    def test_discovery_ignores_database_scripts_without_test_cases(self):
        (self.root / "db_test.py").write_text("print('数据库调试脚本')\n")
        (self.root / "test_api.py").write_text("def test_api():\n    assert 2 > 1\n")
        self.assertEqual(DockerSandbox(self.root).discover_tests(), ["test_api.py"])

    @unittest.skipUnless(os.environ.get("HERMES_SANDBOX_INTEGRATION") == "1", "需要真实 Docker")
    def test_generated_test_runs_in_real_sandbox_before_approval(self):
        client = Client([
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "edit_file", "arguments": {"path": "main.py", "old_text": "x = 1", "new_text": "x = 2"}},
            {"type": "final", "answer": "候选修改已生成"},
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "main.py"}},
            {"type": "tool_call", "tool": "create_file", "arguments": {"path": "tests/test_value.py", "content":
                "from main import x\ndef test_changed_default():\n    assert x == 2\n"}},
            {"type": "final", "answer": "回归测试已生成"},
        ])
        result, proposal = RepairWorkflow(self.root, client, lambda e: None).run("把 x 改成 2", [])
        self.assertTrue(proposal.public()["can_apply"], proposal.validation)
        self.assertIn("1 passed", proposal.validation["output"])
        self.assertFalse((self.root / "tests").exists())
        self.assertEqual((self.root / "main.py").read_text(), "x = 1\n")
        proposal.apply(self.root, self.backups)
        self.assertTrue((self.root / "tests/test_value.py").exists())
        self.assertEqual((self.root / "main.py").read_text(), "x = 2\n")
