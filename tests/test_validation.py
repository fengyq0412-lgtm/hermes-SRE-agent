"""失败反馈、异步测试纠正与源码事实提取的回归测试。"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_sre_agent.repair import ChangeSet, RepairWorkflow
from hermes_sre_agent.code_review import CodeReviewAgent, patch_history
from hermes_sre_agent.sandbox import DockerSandbox
from hermes_sre_agent.model_client import ModelRequestError
from hermes_sre_agent.test_validation import import_isolation_plan, source_contracts, validation_details


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def complete(self, messages):
        self.prompts.append(json.loads(json.dumps(messages)))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return json.dumps(reply)


def call(tool, **arguments):
    return {"type": "tool_call", "tool": tool, "arguments": arguments}


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "main.py").write_text("value = 1\n")

    def replies(self):
        return [call("read_file", path="main.py"),
                call("edit_file", path="main.py", old_text="value = 1", new_text="value = 2"),
                call("create_file", path="tests/test_main.py", content="from main import value\ndef test_value():\n    assert value == 1\n"),
                {"type": "final", "answer": "候选修改与测试已生成"}]

    def test_failed_generated_test_is_corrected_then_revalidated(self):
        client = Client(self.replies() + [
            call("read_file", path="tests/test_main.py"),
            call("edit_file", path="main.py", old_text="value = 2", new_text="value = 1"),
            call("edit_file", path="tests/test_main.py", old_text="assert value == 1", new_text="assert value == 2"),
            {"type": "final", "answer": "已修正测试的预期值"}])
        trace = []
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests", side_effect=[
                {"status": "finished", "exit_code": 1, "output": "assert 2 == 1"},
                {"status": "finished", "exit_code": 0, "output": "1 passed"}]) as run:
            result, proposal = RepairWorkflow(self.root, client, lambda e: None, trace.append).run("将 value 改成 2", [])
        self.assertEqual(run.call_count, 2)
        self.assertTrue(proposal.public()["can_apply"])
        self.assertEqual(proposal.changes["main.py"][1], b"value = 2\n")
        self.assertTrue(any(e["event"] == "patch_retry" for e in trace))
        self.assertIn("assert 2 == 1", json.dumps(client.prompts, ensure_ascii=False))
        self.assertEqual((self.root / "main.py").read_text(), "value = 1\n")
        self.assertEqual(proposal.validation["attempt"], 2)

    def test_unchanged_test_stops_retrying(self):
        client = Client(self.replies() + [call("read_file", path="tests/test_main.py"),
                                        *[{"type": "final", "answer": "当前无法修正"}] * 3])
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests", return_value={
                "status": "finished", "exit_code": 2, "output": "ModuleNotFoundError: No module named 'external_sdk'"}) as run:
            _, proposal = RepairWorkflow(self.root, client, lambda e: None).run("修改", [])
        self.assertEqual(run.call_count, 1)
        self.assertFalse(proposal.public()["can_apply"])
        self.assertEqual(proposal.validation["missing_modules"], ["external_sdk"])

    def test_previous_business_patch_cannot_complete_test_generation(self):
        client = Client([
            call("read_file", path="main.py"),
            call("edit_file", path="main.py", old_text="value = 1", new_text="value = 2"),
            {"type": "final", "answer": "业务补丁已生成"},
            call("read_file", path="main.py"),
            {"type": "final", "answer": "测试文件已经创建并覆盖全部分支"},
            call("create_file", path="tests/test_main.py", content="from main import value\ndef test_value():\n    assert value == 2\n"),
            {"type": "final", "answer": "测试已实际创建"},
        ])
        trace = []
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests", return_value={
                "status": "finished", "exit_code": 0, "output": "1 passed"}) as run:
            _, proposal = RepairWorkflow(self.root, client, lambda e: None, trace.append).run("将 value 改成2", [])
        self.assertTrue(proposal.public()["can_apply"])
        self.assertIn("tests/test_main.py", proposal.changes)
        self.assertEqual(sum(e["event"] == "patch_retry" for e in trace), 1)
        self.assertTrue(any(e["event"] == "test_patch_context" and e["details"]["initial_changed_files"] == 1 for e in trace))
        run.assert_called_once()
        self.assertFalse((self.root / "tests").exists())

    def test_test_stage_budget_still_enters_patch_writer_with_business_changes(self):
        changes = ChangeSet(self.root)
        changes.edit_file("main.py", "value = 1", "value = 2")
        changes.test_only = True
        client = Client([
            call("read_file", path="main.py"),
            {"type": "patch", "changes": [{"path": "tests/test_main.py", "content":
                "from main import value\ndef test_value():\n    assert value == 2\n"}]},
        ])
        trace = []
        CodeReviewAgent(self.root, client, editor=changes, max_tool_calls=1, on_trace=trace.append).run("补充测试")
        self.assertTrue(any(e["event"] == "patch_generation_started" for e in trace))
        self.assertIn("tests/test_main.py", changes.changes())

    def test_missing_test_is_not_reported_as_docker_execution(self):
        client = Client([
            call("read_file", path="main.py"),
            call("edit_file", path="main.py", old_text="value = 1", new_text="value = 2"),
            {"type": "final", "answer": "业务补丁已生成"},
            call("read_file", path="main.py"),
            *[{"type": "final", "answer": "测试已经创建"}] * 3,
        ])
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests") as run:
            _, proposal = RepairWorkflow(self.root, client, lambda e: None).run("修改", [])
        run.assert_not_called()
        self.assertEqual(proposal.validation["status"], "not_run")
        self.assertEqual(proposal.validation["failure_kind"], "tests_not_generated")
        self.assertIn("未启动 Docker", proposal.validation["summary"])
        self.assertFalse(proposal.public()["can_apply"])

    def test_retry_budget_is_three_validation_runs(self):
        replies = self.replies()
        for old, new in ((1, 3), (3, 4)):
            replies += [call("read_file", path="tests/test_main.py"),
                        call("edit_file", path="tests/test_main.py", old_text=f"assert value == {old}", new_text=f"assert value == {new}"),
                        {"type": "final", "answer": "已尝试调整"}]
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests", return_value={
                "status": "finished", "exit_code": 1, "output": "断言失败"}) as run:
            _, proposal = RepairWorkflow(self.root, Client(replies), lambda e: None).run("修改", [])
        self.assertEqual(run.call_count, 3)
        self.assertFalse(proposal.public()["can_apply"])

    def test_source_contracts_include_async_function_and_real_keys(self):
        source = 'async def remove(request):\n    data = await request.json()\n    return response(data["itemId"], "old")\n\ndef response(value, message):\n    return message\n'
        after = source.replace('"old"', '"new"')
        (self.root / "handler.py").write_text(after)
        records = source_contracts(self.root, {"handler.py": (source.encode(), after.encode())})
        definitions = records[0]["definitions"]
        self.assertEqual(definitions[0]["name"], "remove")
        self.assertTrue(definitions[0]["async"])
        self.assertIn('"itemId"', definitions[0]["source"])
        self.assertEqual(definitions[1]["name"], "response")

    def test_dependency_failure_is_not_a_passing_test(self):
        details = validation_details({"status": "finished", "exit_code": 2,
                                      "output": "ModuleNotFoundError: No module named 'fastapi'"})
        self.assertEqual(details["failure_kind"], "dependency_missing")
        self.assertIn("fastapi", details["summary"])

    def test_direct_test_patch_needs_no_final_model_call(self):
        client = Client(self.replies()[:2] + [{"type": "final", "answer": "业务完成"},
            {"type": "patch", "changes": [{"path": "tests/test_main.py", "content":
                "from main import value\ndef test_value():\n    assert value == 2\n"}]}])
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests", return_value={
                "status": "finished", "exit_code": 0, "output": "1 passed"}):
            _, proposal = RepairWorkflow(self.root, client, lambda e: None).run("修改 value", [])
        self.assertTrue(proposal.public()["can_apply"])
        self.assertEqual(len(client.prompts), 4)
        self.assertIn("1: value = 2", client.prompts[-1][-1]["content"])

    def test_new_test_with_empty_old_text_is_created_without_reading_missing_file(self):
        client = Client(self.replies()[:2] + [{"type": "final", "answer": "业务完成"},
            {"type": "patch", "changes": [{"path": "tests/test_main.py", "old_text": "", "new_text":
                "from main import value\ndef test_value():\n    assert value == 2\n"}]}])
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests", return_value={
                "status": "finished", "exit_code": 0, "output": "1 passed"}) as run:
            _, proposal = RepairWorkflow(self.root, client, lambda e: None).run("修改 value", [])
        run.assert_called_once()
        self.assertTrue(proposal.public()["can_apply"])
        self.assertIsNone(proposal.changes["tests/test_main.py"][0])
        self.assertFalse((self.root / "tests").exists())

    def test_missing_file_wrong_protocol_reports_specific_reason_to_ui(self):
        invalid = {"type": "patch", "changes": [{"path": "tests/test_main.py", "start_line": 1,
                    "end_line": 1, "new_text": "def test_value():\n    assert True\n"}]}
        client = Client(self.replies()[:2] + [{"type": "final", "answer": "业务完成"}, invalid, invalid, invalid])
        trace = []
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests") as run:
            _, proposal = RepairWorkflow(self.root, client, lambda e: None, trace.append).run("修改 value", [])
        run.assert_not_called()
        report = proposal.validation["test_agent"]
        self.assertEqual(report["error_kind"], "missing_create_content")
        self.assertIn("新文件", report["summary"])
        self.assertFalse(proposal.public()["can_apply"])
        retries = [e for e in trace if e["event"] == "patch_retry"]
        self.assertEqual(retries[0]["details"]["edit_shapes"][0]["fields"], ["path", "new_text", "start_line", "end_line"])
        self.assertNotIn("assert True", json.dumps(retries))

    def test_malformed_schema_is_not_reported_as_unread_file(self):
        changes = ChangeSet(self.root)
        client = Client([call("read_file", path="main.py"),
                         *[{"type": "patch", "changes": [{"file": "main.py", "code": "value=3"}]}] * 3])
        result = CodeReviewAgent(self.root, client, editor=changes, max_tool_calls=1).run("修改")
        self.assertEqual(result["error_kind"], "invalid_patch_schema")

    def test_empty_old_text_never_overwrites_or_escapes(self):
        changes = ChangeSet(self.root)
        for path in ("main.py", "../outside.py", "/tmp/escape.py"):
            with self.assertRaises(ValueError):
                changes.apply_edits([{"path": path, "old_text": "", "new_text": "value = 9\n"}])
        (self.root / "alias").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            changes.apply_edits([{"path": "alias/test_new.py", "old_text": "", "new_text": "x=1\n"}])
        changes.test_only = True
        with self.assertRaises(ValueError):
            changes.apply_edits([{"path": "other.py", "old_text": "", "new_text": "x=1\n"}])
        self.assertEqual((self.root / "main.py").read_text(), "value = 1\n")
        self.assertEqual(changes.changes(), {})

    def test_scaffold_without_cases_is_not_a_completed_test(self):
        (self.root / "main.py").write_text('from adapter import fetch\nvalue = 1\n')
        (self.root / "adapter.py").write_text('import requests\ndef fetch():\n    return requests.get("https://example.invalid")\n')
        client = Client(self.replies()[:2] + [{"type": "final", "answer": "业务完成"},
                        {"type": "blocked", "reason": "无法提供有效用例"}])
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests") as run:
            _, proposal = RepairWorkflow(self.root, client, lambda e: None).run("修改 value", [])
        run.assert_not_called()
        self.assertIn("tests/test_hermes_main.py", proposal.changes)
        self.assertEqual(proposal.validation["status"], "not_run")
        self.assertEqual(proposal.validation["test_agent"]["outcome"], "blocked")
        self.assertFalse(proposal.public()["can_apply"])
        self.assertFalse((self.root / "tests").exists())

    def test_scaffold_cannot_be_removed_or_bypassed_with_top_level_import(self):
        changes = ChangeSet(self.root)
        name = "tests/test_main.py"
        prefix = "# 保留隔离边界\n"
        changes.create_file(name, prefix)
        changes.test_scaffolds[name] = (prefix, "main")
        for replacement in ("import main\n", prefix + "from main import value\n"):
            with self.assertRaises(ValueError):
                changes.apply_edits([{"path": name, "old_text": prefix, "new_text": replacement}])
            self.assertEqual((self.root / name).read_text(), prefix)

    def test_scaffold_append_requires_current_read_version(self):
        changes = ChangeSet(self.root)
        name = "tests/test_main.py"
        prefix = "# 保留隔离边界\n"
        changes.create_file(name, prefix)
        changes.test_scaffolds[name] = (prefix, "main")
        edit = {"path": name, "append_text": "def test_value():\n    assert 1 == 1\n"}
        with self.assertRaisesRegex(ValueError, "当前版本末尾"):
            changes.apply_edits([edit])
        changes.allow_range(name, 1, 1)
        changes.apply_edits([edit])
        with self.assertRaisesRegex(ValueError, "当前版本末尾"):
            changes.apply_edits([edit])
        with self.assertRaises(ValueError):
            changes.apply_edits([{"path": "main.py", "append_text": "value=3\n"}])

    @unittest.skipUnless(os.environ.get("HERMES_SANDBOX_INTEGRATION") == "1", "需要真实 Docker")
    def test_real_scaffold_with_generated_cases_is_validated(self):
        (self.root / "main.py").write_text('from adapter import fetch\nvalue = 1\n')
        (self.root / "adapter.py").write_text('import requests\ndef fetch():\n    return requests.get("https://example.invalid")\n')
        client = Client(self.replies()[:2] + [{"type": "final", "answer": "业务完成"},
            {"type": "patch", "changes": [{"path": "tests/test_hermes_main.py",
              "append_text": "def test_value(subject):\n    assert subject.value == 2\n"}]}])
        _, proposal = RepairWorkflow(self.root, client, lambda e: None).run("修改 value", [])
        self.assertTrue(proposal.public()["can_apply"], proposal.validation)
        self.assertIn("1 passed", proposal.validation["output"])
        self.assertFalse((self.root / "tests").exists())

    @unittest.skipUnless(os.environ.get("HERMES_SANDBOX_INTEGRATION") == "1", "需要真实 Docker")
    def test_real_sandbox_runs_test_created_with_compatible_protocol(self):
        client = Client(self.replies()[:2] + [{"type": "final", "answer": "业务完成"},
            {"type": "patch", "changes": [{"path": "tests/test_main.py", "old_text": "", "new_text":
                "from main import value\ndef test_value():\n    assert value == 2\n"}]}])
        _, proposal = RepairWorkflow(self.root, client, lambda e: None).run("修改 value", [])
        self.assertTrue(proposal.public()["can_apply"], proposal.validation)
        self.assertIn("1 passed", proposal.validation["output"])
        self.assertFalse((self.root / "tests").exists())

    def test_test_model_timeout_is_reported_without_hiding_sandbox_failure(self):
        failure = ModelRequestError("模拟超时", kind="timeout", retryable=True)
        client = Client(self.replies() + [failure, failure])
        trace = []
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests", return_value={
                "status": "finished", "exit_code": 2, "output": "No module named 'requests'"}) as run:
            _, proposal = RepairWorkflow(self.root, client, lambda e: None, trace.append).run("修改", [])
        run.assert_called_once()
        self.assertEqual(proposal.validation["missing_modules"], ["requests"])
        self.assertEqual(proposal.validation["test_agent"]["error_kind"], "timeout")
        self.assertFalse(proposal.public()["can_apply"])
        self.assertEqual(sum(e["event"] == "test_model_retry" for e in trace), 1)
        self.assertFalse((self.root / "tests").exists())

    def test_transient_test_model_failure_retries_once_and_applies_only_test_patch(self):
        client = Client(self.replies() + [ModelRequestError("网络故障", kind="connection_error", retryable=True),
            {"type": "patch", "changes": [{"path": "tests/test_main.py", "old_text": "value == 1", "new_text": "value == 2"}]}])
        with patch("hermes_sre_agent.repair.DockerSandbox.run_tests", side_effect=[
                {"status": "finished", "exit_code": 1, "output": "assert 2 == 1"},
                {"status": "finished", "exit_code": 0, "output": "1 passed"}]):
            _, proposal = RepairWorkflow(self.root, client, lambda e: None).run("修改", [])
        self.assertTrue(proposal.public()["can_apply"])
        self.assertEqual(proposal.changes["main.py"][1], b"value = 2\n")

    def test_source_contracts_keep_route_status_separate_from_body_code(self):
        before = '@app.post("/remove", response_model=Response)\nasync def remove():\n    return Response(code=400, message="old")\n'
        after = before.replace('"old"', '"new"')
        (self.root / "main.py").write_text(after)
        facts = source_contracts(self.root, {"main.py": (before.encode(), after.encode())})
        definition = facts[0]["definitions"][0]
        self.assertIn('app.post', definition["decorators"][0])
        self.assertNotIn('status_code', definition["decorators"][0])
        self.assertIn('code=400', definition["source"])

    def test_patch_writer_keeps_late_failure_after_large_source(self):
        history = [{"role": "user", "content": "源码" * 20000},
                   {"role": "user", "content": "导入隔离参考"},
                   {"role": "user", "content": "ModuleNotFoundError: No module named 'requests'"}]
        changes = ChangeSet(self.root)
        client = Client([call("read_file", path="main.py"),
                         {"type": "blocked", "reason": "缺少依赖"}])
        CodeReviewAgent(self.root, client, editor=changes, max_tool_calls=1).run("修改", history=history)
        prompt = client.prompts[-1][-1]["content"]
        self.assertIn("No module named 'requests'", prompt)
        self.assertIn("导入隔离参考", prompt)
        packed = json.loads(patch_history(history))
        self.assertEqual(len(packed), 3)
        self.assertLess(len(patch_history(history)), 29000)

    def test_import_plan_uses_real_sync_signatures_and_excludes_changed_module(self):
        (self.root / "main.py").write_text("from adapter import drop, fetch\nfrom changed import save\n")
        (self.root / "adapter.py").write_text("import absent_driver\ndef drop(item):\n    pass\nasync def fetch():\n    pass\n")
        (self.root / "changed.py").write_text("def save():\n    pass\n")
        plans = import_isolation_plan(self.root, {"main.py": (b"", b""), "changed.py": (b"", b"")})
        plan = plans[0]
        self.assertEqual([d["path"] for d in plan["local_dependencies"]], ["adapter.py"])
        self.assertIn("adapter.drop = Mock()", plan["fixture_example"])
        self.assertIn("adapter.fetch = AsyncMock()", plan["fixture_example"])
        self.assertEqual(plan["local_dependencies"][0]["symbols"][0]["line"], 2)
        self.assertNotIn("adapter.save", plan["fixture_example"])
        compile(plan["fixture_example"], "隔离参考", "exec")

    def test_import_plan_does_not_fake_class_or_constant(self):
        (self.root / "main.py").write_text("from adapter import Response, TIMEOUT\n")
        (self.root / "adapter.py").write_text("class Response:\n    pass\nTIMEOUT = 30\n")
        self.assertEqual(import_isolation_plan(self.root, {"main.py": (b"", b"")}), [])

    @unittest.skipUnless(os.environ.get("HERMES_SANDBOX_INTEGRATION") == "1", "需要真实 Docker")
    def test_real_async_plugin_and_import_isolation(self):
        source = '''import logging
from fastapi import FastAPI
from pydantic import BaseModel
from adapter import drop
handler = logging.FileHandler("/missing/log/path/app.log")
app = FastAPI()
class Response(BaseModel):
    code: int
    message: str
async def remove(request):
    data = await request.json()
    drop(data["itemId"])
    return Response(code=200, message="removed")
'''
        (self.root / "main.py").write_text(source)
        (self.root / "adapter.py").write_text("import absent_driver\ndef drop(item):\n    pass\n")
        fixture = import_isolation_plan(self.root, {"main.py": (b"", source.encode())})[0]["fixture_example"]
        tests = fixture + '''
from types import SimpleNamespace
@pytest.mark.asyncio
async def test_remove(subject):
    response = await subject.remove(SimpleNamespace(json=AsyncMock(return_value={"itemId": "one"})))
    assert response.code == 200
    assert response.message == "removed"
    subject.drop.assert_called_once_with("one")
'''
        (self.root / "tests").mkdir()
        path = self.root / "tests/test_main.py"
        path.write_text(tests)
        result = DockerSandbox(self.root).run_tests("tests/test_main.py")
        self.assertEqual(result["exit_code"], 0, result)
        self.assertIn("1 passed", result["output"])
        path.write_text(tests.replace('response.code == 200', 'response.code == 999'))
        failed = DockerSandbox(self.root).run_tests("tests/test_main.py")
        self.assertEqual(failed["exit_code"], 1, failed)
        self.assertIn("1 failed", failed["output"])

    @unittest.skipUnless(os.environ.get("HERMES_SANDBOX_INTEGRATION") == "1", "需要真实 Docker")
    def test_real_fastapi_test_repair_after_collection_failure(self):
        source = '''import asyncio
from fastapi import FastAPI, Request
from pydantic import BaseModel
from external_store import drop
app = FastAPI()
class Response(BaseModel):
    code: int
    message: str
@app.post("/remove")
async def remove_item(request: Request):
    try:
        data = await request.json()
        drop(data["itemId"])
        return Response(code=200, message="updated")
    except KeyError:
        return Response(code=422, message="missing")
    except asyncio.TimeoutError:
        return Response(code=504, message="timeout")
'''
        (self.root / "handler.py").write_text(source)
        (self.root / "external_store.py").write_text("import example_database_driver\n")
        bad = 'from handler import remove_item\ndef test_remove():\n    assert remove_item(None).code == 200\n'
        good = '''import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
import pytest

@pytest.fixture
def handler():
    # 在外部存储适配器边界隔离依赖，目标处理函数和响应模型仍来自真实源码。
    store = ModuleType("external_store")
    store.drop = Mock()
    with patch.dict(sys.modules, {"external_store": store}):
        sys.modules.pop("handler", None)
        yield importlib.import_module("handler")

def test_success(handler):
    request = SimpleNamespace(json=AsyncMock(return_value={"itemId": "one"}))
    response = asyncio.run(handler.remove_item(request))
    assert response.code == 200
    assert response.message == "removed"
    handler.drop.assert_called_once_with("one")

def test_missing_key(handler):
    response = asyncio.run(handler.remove_item(SimpleNamespace(json=AsyncMock(return_value={}))))
    assert response.code == 422
    handler.drop.assert_not_called()

def test_timeout(handler):
    handler.drop.side_effect = asyncio.TimeoutError
    response = asyncio.run(handler.remove_item(SimpleNamespace(json=AsyncMock(return_value={"itemId": "one"}))))
    assert response.code == 504
'''
        client = Client([
            call("read_file", path="handler.py"),
            call("edit_file", path="handler.py", old_text='message="updated"', new_text='message="removed"'),
            call("create_file", path="tests/test_handler.py", content=bad),
            {"type": "final", "answer": "已生成候选补丁"},
            call("read_file", path="tests/test_handler.py"),
            call("edit_file", path="tests/test_handler.py", old_text=bad, new_text=good),
            {"type": "final", "answer": "已隔离外部存储依赖并修正异步测试"},
        ])
        result, proposal = RepairWorkflow(self.root, client, lambda e: None).run("修复删除成功文案", [])
        validations = [s["result"] for s in result["steps"] if s["tool"] == "run_tests"]
        self.assertEqual(validations[0]["failure_kind"], "dependency_missing")
        self.assertTrue(proposal.public()["can_apply"], proposal.validation)
        self.assertIn("3 passed", proposal.validation["output"])
        self.assertEqual(proposal.validation["attempt"], 2)
        self.assertEqual((self.root / "handler.py").read_text(), source)
        self.assertFalse((self.root / "tests").exists())
