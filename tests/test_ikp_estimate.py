import importlib
import json
import os
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
ikp = importlib.import_module("ikp_estimate")


class FakeResponse:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, headers, json):
        self.requests.append({"url": url, "headers": headers, "json": json})
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


class IKPEstimateTransportTests(unittest.TestCase):
    def test_chat_request_and_response_extraction(self):
        client = FakeClient([FakeResponse(200, {
            "choices": [{
                "message": {
                    "content": "Paris",
                    "reasoning": "The answer is probably Paris.",
                }
            }]
        })])
        query = ikp.make_query_fn(
            "https://target.example/v1/", "target-key", "target-model",
            client_factory=lambda timeout: client,
            sleep_fn=lambda seconds: None,
        )

        self.assertEqual(query("What is the capital of France?"), "Paris")
        request = client.requests[0]
        self.assertEqual(request["url"], "https://target.example/v1/chat/completions")
        self.assertEqual(request["headers"]["Authorization"], "Bearer target-key")
        self.assertEqual(request["json"], {
            "model": "target-model",
            "messages": [
                {"role": "system", "content": ikp.SYSTEM_MSG},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "temperature": 0,
        })

    def test_responses_request_and_final_output_extraction(self):
        client = FakeClient([FakeResponse(200, {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Paris"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Paris"}],
                },
            ]
        })])
        query = ikp.make_query_fn(
            "https://opencode.ai/zen/go/v1", "go-key",
            "muse-spark-1.3-contributor", api_style="responses",
            client_factory=lambda timeout: client,
            sleep_fn=lambda seconds: None,
        )

        self.assertEqual(query("What is the capital of France?"), "Paris")
        request = client.requests[0]
        self.assertEqual(
            request["url"], "https://opencode.ai/zen/go/v1/responses"
        )
        self.assertEqual(request["json"], {
            "model": "muse-spark-1.3-contributor",
            "instructions": ikp.SYSTEM_MSG,
            "input": [{
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "What is the capital of France?",
                }],
            }],
        })
        self.assertNotIn("messages", request["json"])
        self.assertNotIn("temperature", request["json"])

    def test_responses_xhigh_reasoning_field(self):
        client = FakeClient([FakeResponse(200, {"output_text": "Paris"})])
        query = ikp.make_query_fn(
            "https://opencode.ai/zen/go/v1", "go-key", "target-model",
            api_style="responses", reasoning_effort="xhigh",
            client_factory=lambda timeout: client,
            sleep_fn=lambda seconds: None,
        )

        self.assertEqual(query("question"), "Paris")
        self.assertEqual(client.requests[0]["json"]["reasoning"],
                         {"effort": "xhigh"})
        self.assertEqual(query.reasoning_effort, "xhigh")

    def test_thinking_flag_remains_medium_compatibility_alias(self):
        endpoint, payload = ikp.build_target_request(
            "https://target.example/v1", "target-model", "question",
            is_thinking=True,
        )
        self.assertEqual(endpoint, "https://target.example/v1/chat/completions")
        self.assertEqual(payload["reasoning"], {"effort": "medium"})

    def test_request_preview_redacts_key_and_is_network_free(self):
        preview = ikp.target_request_preview(
            "https://opencode.ai/zen/go/v1", "target-model",
            api_style="responses", reasoning_effort="xhigh", has_api_key=True,
        )

        serialized = json.dumps(preview)
        self.assertNotIn("go-key", serialized)
        self.assertEqual(preview["headers"]["Authorization"], "Bearer <redacted>")
        self.assertEqual(preview["payload"]["reasoning"], {"effort": "xhigh"})

    def test_opencode_target_key_is_separate_from_judge_key(self):
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "judge-key",
            "OPENCODE_GO_API_KEY": "target-key",
        }, clear=True):
            self.assertEqual(
                ikp.resolve_target_api_key(
                    "https://opencode.ai/zen/go/v1", None
                ),
                "target-key",
            )

    def test_target_api_failure_is_not_a_refusal(self):
        client = FakeClient([
            FakeResponse(503, {}),
            FakeResponse(503, {}),
            FakeResponse(503, {}),
        ])
        query = ikp.make_query_fn(
            "https://target.example/v1", "target-key", "target-model",
            client_factory=lambda timeout: client,
            sleep_fn=lambda seconds: None,
        )

        with self.assertRaises(ikp.TargetAPIError):
            query("question")
        self.assertEqual(len(client.requests), 3)

    def test_judge_api_failure_is_not_wrong(self):
        client = FakeClient([
            FakeResponse(500, {}),
            FakeResponse(500, {}),
            FakeResponse(500, {}),
        ])
        judge = ikp.make_judge_fn(
            "judge-key", client_factory=lambda timeout: client,
            sleep_fn=lambda seconds: None,
        )

        with self.assertRaises(ikp.JudgeAPIError):
            judge("question", "gold", "answer")
        self.assertEqual(len(client.requests), 3)

    def test_judge_success_keeps_fixed_request_and_scoring(self):
        client = FakeClient([FakeResponse(200, {
            "choices": [{"message": {"content": "CORRECT"}}]
        })])
        judge = ikp.make_judge_fn(
            "judge-key", client_factory=lambda timeout: client,
            sleep_fn=lambda seconds: None,
        )

        self.assertEqual(judge("question", "gold", "answer"), "CORRECT")
        request = client.requests[0]
        self.assertEqual(request["url"],
                         "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(request["json"]["model"], ikp.JUDGE_MODEL)
        self.assertEqual(request["json"]["temperature"], 0)
        self.assertEqual(request["json"]["reasoning"], {"effort": "low"})
        self.assertIn("Reply one word: CORRECT, REFUSAL, or WRONG",
                      request["json"]["messages"][0]["content"])


class IKPSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probes = [
            {"id": f"{tier}-{index}", "tier": tier}
            for tier in ikp.TIERS
            for index in range(50)
        ]

    def test_sample_10_is_exact_and_balanced(self):
        selected = ikp.stratified_sample(self.probes, 10, seed=42)
        counts = Counter(probe["tier"] for probe in selected)
        self.assertEqual(len(selected), 10)
        self.assertEqual([counts[tier] for tier in ikp.TIERS],
                         [2, 2, 2, 1, 1, 1, 1])

    def test_sample_200_is_exact_and_balanced(self):
        selected = ikp.stratified_sample(self.probes, 200, seed=42)
        counts = Counter(probe["tier"] for probe in selected)
        self.assertEqual(len(selected), 200)
        self.assertEqual([counts[tier] for tier in ikp.TIERS],
                         [29, 29, 29, 29, 28, 28, 28])

    def test_sampling_is_deterministic_for_a_seed(self):
        first = [probe["id"] for probe in ikp.stratified_sample(self.probes, 200, 42)]
        second = [probe["id"] for probe in ikp.stratified_sample(self.probes, 200, 42)]
        different = [probe["id"] for probe in ikp.stratified_sample(self.probes, 200, 43)]
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)


if __name__ == "__main__":
    unittest.main()
