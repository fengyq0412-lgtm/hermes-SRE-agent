"""V0.2 运维写入边界：计划、一次性审批、冲突保护和变更后验证。"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from hermes_sre_agent.remediation import ChangeManager, RemediationError


class FakeAdapter:
    def __init__(self):
        self.identity = "a" * 64
        self.restarts = []
        self.health_result = {"status_code": 200, "latency_ms": 100,
                              "body": {"batch_size": 16}}

    def container_id(self, _name):
        return self.identity

    def restart(self, name, expected_id):
        self.restarts.append((name, expected_id))

    def health(self, _url):
        return self.health_result


class RemediationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "state.json"
        self.config.write_text('{"batch_size":64,"untouched":"keep"}\n')
        self.policy = self.root / "policy.json"
        self.policy.write_text(json.dumps({"version": 1, "services": {"service-a": {
            "config_file": "state.json", "container": "service-a",
            "health_url": "http://127.0.0.1:8000/health", "max_latency_ms": 1000,
            "verify_fields": {"batch_size": "batch_size"},
            "allowed_keys": {"batch_size": {"type": "integer", "min": 1, "max": 64}}
        }}}))
        self.adapter = FakeAdapter()
        self.manager = ChangeManager(self.policy, self.root / "private", self.adapter)

    def test_one_time_token_is_bound_to_plan_and_restarts_then_verifies(self):
        plan = self.manager.prepare("service-a", {"batch_size": 16})
        self.assertEqual(self.config.read_text(), '{"batch_size":64,"untouched":"keep"}\n')
        with self.assertRaisesRegex(RemediationError, "确认不一致"):
            self.manager.approve(plan["id"], "other-service")
        token = self.manager.approve(plan["id"], "service-a")
        with self.assertRaisesRegex(RemediationError, "令牌无效"):
            self.manager.apply(plan["id"], "wrong")
        result = self.manager.apply(plan["id"], token)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(self.adapter.restarts, [("service-a", "a" * 64)])
        self.assertEqual(json.loads(self.config.read_text()), {"batch_size": 16, "untouched": "keep"})
        self.assertEqual(Path(result["backup_path"]).read_text(), '{"batch_size":64,"untouched":"keep"}\n')
        self.assertEqual(self.manager.show(plan["id"])["status"], "verified")
        with self.assertRaisesRegex(RemediationError, "令牌无效"):
            self.manager.apply(plan["id"], token)

    def test_stale_config_or_container_consumes_token_without_writing(self):
        plan = self.manager.prepare("service-a", {"batch_size": 16})
        token = self.manager.approve(plan["id"], "service-a")
        self.config.write_text('{"batch_size":32,"untouched":"user"}\n')
        result = self.manager.apply(plan["id"], token)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stage"], "preflight")
        self.assertFalse(self.adapter.restarts)
        self.assertEqual(json.loads(self.config.read_text())["batch_size"], 32)
        with self.assertRaises(RemediationError):
            self.manager.apply(plan["id"], token)

        second = self.manager.prepare("service-a", {"batch_size": 16})
        token = self.manager.approve(second["id"], "service-a")
        self.adapter.identity = "b" * 64
        result = self.manager.apply(second["id"], token)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(self.adapter.restarts)

    def test_allowlist_and_verification_failure(self):
        for changes in ({"untouched": "erase"}, {"batch_size": 100},
                        {"api_key": "secret"}, {"batch_size": True}):
            with self.subTest(changes=changes), self.assertRaises(RemediationError):
                self.manager.prepare("service-a", changes)
        plan = self.manager.prepare("service-a", {"batch_size": 16})
        token = self.manager.approve(plan["id"], "service-a")
        self.adapter.health_result = {"status_code": 500, "latency_ms": 20, "body": {}}
        result = self.manager.apply(plan["id"], token)
        self.assertEqual(result["status"], "verification_failed")
        self.assertEqual(len(result["observations"]), 3)
        self.assertTrue(Path(result["backup_path"]).exists())
        self.assertEqual(json.loads(self.config.read_text())["batch_size"], 16)

    def test_expired_token_and_changed_policy_do_not_write_or_restart(self):
        plan = self.manager.prepare("service-a", {"batch_size": 16})
        token = self.manager.approve(plan["id"], "service-a")
        with sqlite3.connect(self.manager.db_path) as db:
            db.execute("UPDATE changes SET token_expires=0 WHERE id=?", (plan["id"],))
        with self.assertRaisesRegex(RemediationError, "已过期"):
            self.manager.apply(plan["id"], token)
        self.assertFalse(self.adapter.restarts)
        self.assertEqual(json.loads(self.config.read_text())["batch_size"], 64)

        second = self.manager.prepare("service-a", {"batch_size": 16})
        token = self.manager.approve(second["id"], "service-a")
        self.policy.write_text(self.policy.read_text() + "\n")
        result = self.manager.apply(second["id"], token)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stage"], "preflight")
        self.assertFalse(self.adapter.restarts)

    def test_demo_policy_can_restore_normal_state_and_log(self):
        from demo.inject_fault import FAULTS

        policy_path = Path(__file__).resolve().parents[1] / "demo/change-policy.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        allowed = policy["services"]["legal-agent"]["allowed_keys"]
        normal = {"scenario": "normal", **FAULTS["normal"]}
        ChangeManager._validate_changes(normal, allowed)
        self.assertEqual(allowed["log_message"]["choices"], [normal["log_message"]])
