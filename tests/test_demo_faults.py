import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DemoFaultTests(unittest.TestCase):
    def test_fault_injector_writes_a_reproducible_slow_api_state(self):
        root = Path(__file__).resolve().parents[1]
        injector = root / "demo" / "inject_fault.py"
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(injector), "--scenario", "api_slow", "--runtime-dir", directory],
                text=True, capture_output=True, check=True,
            )
            state = json.loads((Path(directory) / "state.json").read_text(encoding="utf-8"))

        self.assertIn("已注入 api_slow", completed.stdout)
        self.assertEqual(state["status_code"], 200)
        self.assertEqual(state["delay_seconds"], 4.2)
        self.assertEqual(state["batch_size"], 64)
