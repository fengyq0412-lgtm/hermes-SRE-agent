"""验证会话恢复、上下文压缩、历史检索和项目隔离。"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hermes_sre_agent.conversation import ContextMemory, ConversationStore
from hermes_sre_agent.model_client import ModelRequestError
from hermes_sre_agent.web_server import ProjectRegistry, ReviewJobs


def turn(index, question=None, answer=None):
    return {"id": str(index), "question": question or f"第 {index} 轮问题", "answer": answer or f"第 {index} 轮回答",
            "status": "completed", "memory": {}}


class ContextTests(unittest.TestCase):
    def test_summary_keeps_old_constraint_and_recent_window(self):
        turns = [turn(1, "只修改业务代码，不要改变接口返回格式"), turn(2), turn(3), turn(4)]
        calls = []
        def complete(messages):
            calls.append(messages)
            return json.dumps({"summary": "用户要求只修改业务代码，不要改变接口返回格式。"}, ensure_ascii=False)
        history, memory, info = ContextMemory().prepare(turns, SimpleNamespace(complete=complete))
        self.assertIn("不要改变接口返回格式", history[0]["content"])
        self.assertEqual([m["content"] for m in history if m["role"] == "user"][1:], [t["question"] for t in turns[1:]])
        self.assertEqual(info["compressed_turns"], 1)
        self.assertEqual(info["recent_turns"], 3)
        self.assertEqual(memory["method"], "model")
        self.assertEqual(len(calls), 1)
        self.assertIn(turns[0]["question"], calls[0][-1]["content"])

    def test_incremental_compression_only_adds_newly_evicted_turn(self):
        turns = [turn(i) for i in range(1, 6)]
        turns[-1]["memory"] = {"summary": "第一轮的旧约束", "compressed_turns": 1, "through_id": "1"}
        captured = []
        def complete(messages):
            captured.append(messages[-1]["content"])
            return '{"summary":"第一轮的旧约束，第二轮的目标"}'
        _, memory, _ = ContextMemory().prepare(turns, SimpleNamespace(complete=complete))
        self.assertIn("第一轮的旧约束", captured[0])
        self.assertIn("第 2 轮问题", captured[0])
        self.assertNotIn("第 1 轮问题", captured[0])
        self.assertNotIn("第 3 轮问题", captured[0])
        self.assertEqual(memory["through_id"], "2")

    def test_failed_compression_falls_back_and_respects_character_budget(self):
        turns = [turn(i, "约束" * 2000, "分析" * 16000) for i in range(5)]
        def fail(_messages):
            raise ModelRequestError("模拟摘要服务失败")
        history, _, info = ContextMemory(max_chars=8000).prepare(turns, SimpleNamespace(complete=fail))
        self.assertLessEqual(sum(len(m["content"]) for m in history), 8000)
        self.assertEqual(info["summary_method"], "extractive")
        self.assertGreater(info["shortened_messages"], 0)

    def test_archive_search_recovers_detail_outside_window(self):
        turns = [turn(1, "校验规则：只有文档创建人能删除分段"), *[turn(i) for i in range(2, 10)]]
        result = ContextMemory.search(turns, "创建人")
        self.assertEqual(result["matches"][0]["turn_id"], "1")
        self.assertEqual(result["matches"][0]["source"], "question")
        self.assertIn("只有文档创建人", result["matches"][0]["excerpt"])

    def test_router_receives_old_summary_as_well_as_last_exchange(self):
        history = [{"role": "user", "content": "较早约定：只允许创建人删除分段"},
                   {"role": "user", "content": "解释错误处理"}, {"role": "assistant", "content": "解释完毕"}]
        captured = []
        def complete(messages):
            captured.extend(messages)
            return '{"mode":"repair"}'
        ReviewJobs._choose_mode(SimpleNamespace(complete=complete), "按之前约定加校验", history)
        self.assertIn(history[0], captured)


class PersistenceTests(unittest.TestCase):
    def test_long_conversation_uses_summary_counts_usage_and_can_search_original(self):
        class Client:
            last_usage = None
            def complete(self, _messages):
                self.last_usage = {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42}
                return '{"summary":"用户指定只有创建人可以删除，不得改变接口格式。"}'

        captured = []
        def review(agent, question, history):
            captured.append((history, agent.tools.call("search_history", {"text": "创建人"})))
            return {"answer": "已理解：" + question, "steps": []}

        with tempfile.TemporaryDirectory() as directory:
            registry = ProjectRegistry()
            project_id = registry.add(directory)["id"]
            jobs = ReviewJobs(registry, client_factory=Client)
            try:
                parent = None
                with patch("hermes_sre_agent.web_server.CodeReviewAgent.run", review):
                    for index in range(5):
                        question = "只有创建人能删除，不得改变接口格式" if index == 0 else f"继续讨论第 {index} 个问题"
                        parent = jobs.create(project_id, question, parent_id=parent)["id"]
                        for _ in range(100):
                            if jobs.get(parent)["status"] not in {"queued", "running"}:
                                break
                            time.sleep(.01)
                        self.assertEqual(jobs.get(parent)["status"], "completed")
                job = jobs.get(parent)
                self.assertEqual(job["context"]["compressed_turns"], 1)
                self.assertEqual(job["context"]["summary_method"], "model")
                self.assertEqual(job["usage"]["total_tokens"], 42)
                self.assertIn("只有创建人", captured[-1][0][0]["content"])
                self.assertIn("创建人", captured[-1][1]["matches"][0]["excerpt"])
            finally:
                jobs.executor.shutdown(wait=True)

    def test_restart_restores_turns_and_followup_without_cross_project_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "second").mkdir()
            registry = ProjectRegistry()
            first = registry.add(directory)["id"]
            second = registry.add(str(root / "second"))["id"]
            database = root / "memory.sqlite3"
            jobs = ReviewJobs(registry, client_factory=lambda: object(), conversation_path=database)
            with patch("hermes_sre_agent.web_server.CodeReviewAgent.run", return_value={"answer": "按创建人身份校验", "steps": []}):
                initial = jobs.create(first, "只有创建人能删除")
                jobs.executor.shutdown(wait=True)
            restored = ReviewJobs(registry, client_factory=lambda: object(), conversation_path=database)
            self.addCleanup(restored.executor.shutdown)
            conversation = restored.get_conversation(first, initial["conversation_id"])
            self.assertEqual(conversation["turns"][0]["question"], "只有创建人能删除")
            with patch.object(restored.executor, "submit"):
                followup = restored.create(first, "修改，加上校验", parent_id=initial["id"])
                self.assertEqual(followup["conversation_id"], initial["conversation_id"])
                history = restored.jobs[followup["id"]]["history"]
                self.assertIn("只有创建人能删除", history[0]["content"])
                fresh = restored.create(second, "检查")
                self.assertEqual(restored.jobs[fresh["id"]]["history"], [])
            with self.assertRaises(KeyError):
                restored.get_conversation(second, initial["conversation_id"])
            self.assertEqual(restored.list_conversations(second)[0]["id"], fresh["conversation_id"])

    def test_restart_expires_approval_and_saves_only_source_indexes(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "memory.sqlite3"
            registry = ProjectRegistry()
            project = registry.add(directory)["id"]
            store = ConversationStore(database)
            store.save({"id": "pending", "conversation_id": "conv", "project_id": project, "created_at": 1,
                        "question": "修改", "answer": "待审核", "status": "awaiting_review", "usage": {}, "trace": [],
                        "proposal": {"id": "patch", "state": "pending", "can_apply": True, "files": []},
                        "steps": [{"tool": "read_file", "result": {"path": "main.py", "start_line": 2, "content": "SOURCE_NOT_TO_PERSIST"}}]})
            jobs = ReviewJobs(registry, conversation_path=database)
            self.addCleanup(jobs.executor.shutdown)
            job = jobs.get("pending")
            self.assertEqual(job["status"], "interrupted")
            self.assertFalse(job["proposal"]["can_apply"])
            self.assertEqual(job["proposal"]["state"], "expired")
            self.assertEqual(job["evidence"][0]["path"], "main.py")
            self.assertNotIn("SOURCE_NOT_TO_PERSIST", json.dumps(store.load()))
            with self.assertRaises(ValueError):
                jobs.decide("pending", "patch", "approve")
