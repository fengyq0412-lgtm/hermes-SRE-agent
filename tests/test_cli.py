import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_sre_agent.cli import run_interactive


class InteractiveCliTests(unittest.TestCase):
    def test_interactive_mode_handles_a_question_then_exits(self):
        inputs = iter(["legal-agent 最近很慢", "exit"])

        with tempfile.TemporaryDirectory() as directory, patch("builtins.print") as output:
            run_interactive(
                "scenario", "api_slow", "legal-agent", Path(directory),
                mode="rules", read_input=lambda _: next(inputs),
            )

            self.assertTrue(list(Path(directory).glob("*-report.md")))
            self.assertTrue(list(Path(directory).glob("*-trajectory.json")))
            self.assertTrue(any("已退出 Hermes" in str(call) for call in output.call_args_list))
