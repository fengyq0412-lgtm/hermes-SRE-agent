import json
import os
import unittest
from unittest.mock import patch

from hermes_sre_agent.web_research import WebResearchTools, _fetch_public_https


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self, *_):
        return self.payload


class WebResearchTests(unittest.TestCase):
    def test_search_then_open_page(self):
        tools = WebResearchTools()
        payload = {"web": {"results": [
            {"title": "Docs", "url": "https://example.com/docs", "description": "Official guide"},
            {"title": "Local", "url": "http://127.0.0.1/private", "description": "skip"},
        ]}}
        with patch.dict(os.environ, {"HERMES_BRAVE_API_KEY": "dummy"}), patch(
            "hermes_sre_agent.web_research.urlopen", return_value=_Response(json.dumps(payload).encode())
        ) as request:
            result = tools.call("search_web", {"query": "guide"})
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(request.call_args.args[0].get_header("X-subscription-token"), "dummy")
        with patch("hermes_sre_agent.web_research._fetch_public_https",
                   return_value=("<h1>Guide</h1><script>ignore me</script><p>Useful text</p>", "text/html")):
            page = tools.call("open_page", {"url": "https://example.com/docs"})
        self.assertIn("Useful text", page["content"])
        self.assertNotIn("ignore me", page["content"])

    def test_page_must_be_search_result_and_public_https(self):
        tools = WebResearchTools()
        with self.assertRaises(ValueError):
            tools.call("open_page", {"url": "https://example.com/unsearched"})
        with self.assertRaises(ValueError):
            _fetch_public_https("http://example.com")
        with self.assertRaises(ValueError):
            _fetch_public_https("https://127.0.0.1/private")

    def test_missing_key_is_reported(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "HERMES_BRAVE_API_KEY"):
            WebResearchTools().call("search_web", {"query": "anything"})
