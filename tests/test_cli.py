import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from hermes_sre_agent.cli import main, run_diagnosis, run_interactive


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

    def test_live_mode_rejects_implicit_demo_targets_before_collection(self):
        cases = [
            (["--backend", "live", "--mode", "rules"], "--service"),
            (["--backend", "live", "--mode", "rules", "--service", "hermes-web"], "--container"),
            (["--backend", "live", "--mode", "rules", "--service", "hermes-web",
              "--container", "hermes-web"], "--health-url"),
            (["--backend", "live", "--history"], "--service"),
            (["--backend", "live", "--interactive"], "--service"),
        ]
        for flags, missing in cases:
            with self.subTest(flags=flags), patch("hermes_sre_agent.cli.sys.argv", ["hermes", "诊断"] + flags), \
                 patch("hermes_sre_agent.cli.load_dotenv"), \
                 patch("hermes_sre_agent.cli.run_diagnosis") as diagnose, redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as stopped:
                    main()
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn(missing, error.getvalue())
                diagnose.assert_not_called()

    def test_live_mode_passes_only_explicit_targets(self):
        flags = ["hermes 慢", "--backend", "live", "--mode", "rules", "--service", "hermes-web",
                 "--container", "hermes-api", "--health-url", "http://127.0.0.1:8765/health"]
        with patch("hermes_sre_agent.cli.sys.argv", ["hermes"] + flags), \
             patch("hermes_sre_agent.cli.load_dotenv"), \
             patch("hermes_sre_agent.cli.run_diagnosis") as diagnose:
            self.assertEqual(main(), 0)
        self.assertEqual(diagnose.call_args.args[3], "hermes-web")
        self.assertEqual(diagnose.call_args.args[5:7], ("http://127.0.0.1:8765/health", "hermes-api"))

    def test_scenario_keeps_demo_service_default(self):
        with patch("hermes_sre_agent.cli.sys.argv", ["hermes", "演示", "--mode", "rules"]), \
             patch("hermes_sre_agent.cli.load_dotenv"), \
             patch("hermes_sre_agent.cli.run_diagnosis") as diagnose:
            self.assertEqual(main(), 0)
        self.assertEqual(diagnose.call_args.args[3], "legal-agent")
        self.assertEqual(diagnose.call_args.args[5:7], (None, None))

    def test_programmatic_live_entry_rejects_missing_targets(self):
        with self.assertRaisesRegex(ValueError, "--container"):
            run_diagnosis("诊断", "live", "api_slow", "hermes-web", Path("artifacts"),
                          health_url="http://127.0.0.1:8765/health", mode="rules")
