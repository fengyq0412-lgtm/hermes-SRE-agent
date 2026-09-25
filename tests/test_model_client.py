"""验证兼容模型接口返回的 token 用量只按真实 usage 统计。"""

import json
import unittest
from unittest.mock import patch

from hermes_sre_agent.model_client import ModelConfig, OpenAICompatibleClient


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.body).encode("utf-8")


class ModelUsageTests(unittest.TestCase):
    def test_each_response_exposes_reported_usage_and_clears_missing_usage(self):
        responses = [
            FakeResponse({"choices": [{"message": {"content": "你好"}}],
                          "usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18}}),
            FakeResponse({"choices": [{"message": {"content": "继续"}}]}),
        ]
        client = OpenAICompatibleClient(ModelConfig("https://example.com/v1", "key", "model"))
        with patch("hermes_sre_agent.model_client.urlopen", side_effect=responses):
            self.assertEqual(client.complete([]), "你好")
            self.assertEqual(client.last_usage["total_tokens"], 18)
            self.assertEqual(client.complete([]), "继续")
            self.assertIsNone(client.last_usage)

    def test_compatible_usage_aliases_and_invalid_values(self):
        parse = OpenAICompatibleClient._parse_usage
        self.assertEqual(parse({"usage": {"input_tokens": 7, "output_tokens": 3}}),
                         {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10})
        self.assertIsNone(parse({"usage": {"prompt_tokens": True, "completion_tokens": 3}}))
        self.assertIsNone(parse({"usage": {"prompt_tokens": -1, "completion_tokens": 3}}))

    def test_output_limit_and_finish_reason_are_exposed(self):
        response = FakeResponse({"choices": [{"message": {"content": "{\"type\":\"patch\""},
                                               "finish_reason": "length"}]})
        client = OpenAICompatibleClient(ModelConfig("https://example.com/v1", "key", "model"))
        with patch("hermes_sre_agent.model_client.urlopen", return_value=response) as urlopen:
            client.complete([])
        sent = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(sent["max_tokens"], 4096)
        self.assertEqual(client.last_finish_reason, "length")
