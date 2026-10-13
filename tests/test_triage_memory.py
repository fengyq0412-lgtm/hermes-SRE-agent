"""V0.4 角色分诊与事件记忆的证据边界。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_sre_agent.backends import ScenarioBackend
from hermes_sre_agent.cli import run_diagnosis
from hermes_sre_agent.incident_memory import IncidentMemory
from hermes_sre_agent.triage import TriageAgent


class TriageMemoryTests(unittest.TestCase):
    def test_specialists_share_read_only_evidence_and_coordinator_uses_it(self):
        run = TriageAgent(ScenarioBackend("api_slow")).investigate("检查接口延迟")
        signals = {item["role"]: item["signal"] for item in run.assessments}
        self.assertEqual(signals["可用性"], "available")
        self.assertEqual(signals["性能"], "gpu_pressure")
        self.assertEqual(signals["变更"], "batch_change")
        self.assertIn("batch_size 从 16 调整为 64", run.diagnosis.root_cause)
        self.assertEqual(len(run.calls), len({call.tool for call in run.calls}))
        self.assertTrue(all(call.read_only for call in run.calls))
        successful = {call.tool for call in run.calls if call.error is None}
        self.assertTrue(all(set(item["evidence_tools"]).issubset(successful) for item in run.assessments))

    def test_memory_is_isolated_by_service_and_source_and_does_not_store_raw_input(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = IncidentMemory(Path(directory))
            first = TriageAgent(ScenarioBackend("api_500")).investigate("super-secret incident text")
            memory.record(first, Path(directory) / "first-report.md", "scenario")
            second = TriageAgent(ScenarioBackend("api_500")).investigate("同类故障")
            original_cause = second.diagnosis.root_cause
            related = memory.find_related(second, "scenario")
            self.assertEqual([item["incident_id"] for item in related], [first.incident.incident_id])
            self.assertEqual(second.diagnosis.root_cause, original_cause)
            self.assertEqual(memory.find_related(second, "live"), [])
            second.incident.service = "other-service"
            self.assertEqual(memory.find_related(second, "scenario"), [])
            self.assertNotIn(b"super-secret incident text", memory.path.read_bytes())
            self.assertEqual(memory.history("legal-agent", "scenario")[0]["source"], "scenario")

    def test_cli_report_links_prior_incident_and_no_memory_skips_recording(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            output_dir = Path(directory)
            run_diagnosis("第一次诊断", "scenario", "api_slow", "legal-agent", output_dir, mode="triage")
            run_diagnosis("第二次诊断", "scenario", "api_slow", "legal-agent", output_dir, mode="triage")
            reports = sorted(output_dir.glob("*-report.md"))
            trajectories = [json.loads(path.read_text(encoding="utf-8"))
                            for path in output_dir.glob("*-trajectory.json")]
            self.assertEqual(len(reports), 2)
            self.assertEqual(sorted(len(item["related_incidents"]) for item in trajectories), [0, 1])
            self.assertTrue(any("# 相关历史事件" in path.read_text(encoding="utf-8") for path in reports))
            run_diagnosis("不记录本次", "scenario", "api_slow", "legal-agent", output_dir,
                          mode="triage", remember=False)
            self.assertEqual(len(IncidentMemory(output_dir).history("legal-agent", "scenario")), 2)
