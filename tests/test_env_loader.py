import os
import tempfile
import unittest
from pathlib import Path

from hermes_sre_agent.env_loader import load_dotenv


class DotenvTests(unittest.TestCase):
    def test_loader_reads_quotes_and_does_not_override_existing_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "# Hermes 本地配置\nHERMES_TEST_URL=\"https://example.test/v1\"\nHERMES_TEST_KEY='from-file'\n",
                encoding="utf-8",
            )
            old_url = os.environ.pop("HERMES_TEST_URL", None)
            old_key = os.environ.get("HERMES_TEST_KEY")
            os.environ["HERMES_TEST_KEY"] = "from-system"
            try:
                self.assertEqual(load_dotenv(path), path)
                self.assertEqual(os.environ["HERMES_TEST_URL"], "https://example.test/v1")
                self.assertEqual(os.environ["HERMES_TEST_KEY"], "from-system")
            finally:
                os.environ.pop("HERMES_TEST_URL", None)
                if old_url is not None:
                    os.environ["HERMES_TEST_URL"] = old_url
                if old_key is None:
                    os.environ.pop("HERMES_TEST_KEY", None)
                else:
                    os.environ["HERMES_TEST_KEY"] = old_key
