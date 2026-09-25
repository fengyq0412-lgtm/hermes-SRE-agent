"""快照边界和可选的真实 Docker 隔离验证。"""
import os
import tempfile
import unittest
from pathlib import Path

from hermes_sre_agent.code_review import SourceTools
from hermes_sre_agent.sandbox import DockerSandbox


class SandboxTests(unittest.TestCase):
    def test_snapshot_excludes_secrets_and_symlinks(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as copy:
            project = Path(root)
            (project / "test_main.py").write_text("x = 1")
            (project / ".env").write_text("KEY=secret")
            (project / "credentials.json").write_text("{}")
            (project / "link.py").symlink_to(project / "test_main.py")
            self.assertEqual(DockerSandbox(root).snapshot(copy), 1)
            self.assertEqual([p.name for p in Path(copy).iterdir()], ["test_main.py"])

    def test_test_tool_requires_explicit_enablement(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "未开启"):
                SourceTools(root).call("run_tests", {})
            with self.assertRaises(ValueError):
                DockerSandbox(root).run_tests("../outside")


@unittest.skipUnless(os.environ.get("HERMES_SANDBOX_INTEGRATION") == "1", "需要显式开启真实 Docker 测试")
class DockerIntegrationTests(unittest.TestCase):
    def test_multiple_targets_and_all_skipped_cannot_mask_failure(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "test_first.py").write_text("def test_first():\n    assert 2 > 1\n")
            second = Path(root) / "test_second.py"
            second.write_text("def test_second():\n    assert False\n")
            result = DockerSandbox(root).run_tests(["test_first.py", "test_second.py"])
            self.assertEqual(result["exit_code"], 1, result)
            second.write_text("import pytest\ndef test_second():\n    pytest.skip('未具备执行条件')\n")
            result = DockerSandbox(root).run_tests("test_second.py")
            self.assertEqual(result["exit_code"], 5, result)

    def test_real_container_isolation_and_project_preservation(self):
        source = '''import os
from pathlib import Path
import socket
import pytest

def test_boundaries():
    assert os.getuid() == 65534
    assert not Path(".env").exists()
    assert "HERMES_API_KEY" not in os.environ
    with pytest.raises(OSError):
        Path("/source/test_boundary.py").write_text("changed")
    with pytest.raises(OSError):
        Path("/outside.txt").write_text("changed")
    with socket.socket() as connection:
        connection.settimeout(1)
        with pytest.raises(OSError):
            connection.connect(("1.1.1.1", 443))
    Path("local-result.txt").write_text("allowed in temporary copy")
'''
        with tempfile.TemporaryDirectory() as root:
            project = Path(root)
            (project / "test_boundary.py").write_text(source)
            (project / ".env").write_text("HERMES_API_KEY=not-for-container")
            result = DockerSandbox(root).run_tests("test_boundary.py")
            self.assertEqual(result["exit_code"], 0, result)
            self.assertIn("1 passed", result["output"])
            self.assertFalse((project / "local-result.txt").exists())
            self.assertEqual((project / "test_boundary.py").read_text(), source)

    def test_timeout_stops_execution(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "test_slow.py").write_text("import time\ndef test_slow():\n    time.sleep(30)\n")
            result = DockerSandbox(root, timeout=2).run_tests("test_slow.py")
            self.assertEqual(result["status"], "timeout")

    def test_failed_test_and_output_limit_are_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as root:
            test = Path(root) / "test_failure.py"
            test.write_text("def test_failure():\n    assert False\n")
            result = DockerSandbox(root).run_tests("test_failure.py")
            self.assertEqual(result["status"], "finished")
            self.assertEqual(result["exit_code"], 1, result)
            test.write_text("def test_failure():\n    print('x' * 100000)\n    assert False\n")
            result = DockerSandbox(root).run_tests("test_failure.py")
            self.assertEqual(result["status"], "output_limit")
            self.assertLessEqual(len(result["output"].encode()), 32000)
