import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
ikp_v2 = importlib.import_module("ikp_estimate_v2")


class IKPEstimateV2LiveTests(unittest.TestCase):
    def live_args(self, **overrides):
        values = {
            "api_base": "https://opencode.ai/zen/go/v1",
            "api_key": None,
            "model": "muse-spark-1.3-contributor",
            "api_style": "responses",
            "reasoning_effort": "xhigh",
            "thinking": False,
            "split": "all",
            "workers": 1,
            "sequential": True,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @contextlib.contextmanager
    def one_probe_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probes.json"
            path.write_text(json.dumps([{
                "id": "probe-1",
                "tier": "T1",
                "question": "What is the capital of France?",
                "answer": "Paris",
            }]))
            with patch.object(ikp_v2.v1, "PROBE_FILE", path):
                yield path

    def test_v2_uses_opencode_target_key_and_passes_responses_xhigh(self):
        query_fn = Mock(return_value="Paris")
        judge_fn = Mock(return_value="CORRECT")
        args = self.live_args()

        with self.one_probe_file(), patch.dict(os.environ, {
            "OPENCODE_GO_API_KEY": "target-key",
            "OPENROUTER_API_KEY": "judge-key",
        }, clear=True), patch.object(
            ikp_v2.v1, "resolve_target_api_key",
            wraps=ikp_v2.v1.resolve_target_api_key,
        ) as resolve_key, patch.object(
            ikp_v2.v1, "make_query_fn", return_value=query_fn
        ) as make_query, patch.object(
            ikp_v2.v1, "make_judge_fn", return_value=judge_fn
        ) as make_judge, patch.object(
            ikp_v2, "show"
        ), patch.object(ikp_v2, "robust_estimate", return_value={}):
            ikp_v2.run_live(args)

        resolve_key.assert_called_once_with(args.api_base, None)
        make_query.assert_called_once_with(
            args.api_base,
            "target-key",
            args.model,
            is_thinking=False,
            api_style="responses",
            reasoning_effort="xhigh",
        )
        make_judge.assert_called_once_with("judge-key")
        self.assertNotEqual(make_query.call_args.args[1], "judge-key")
        self.assertNotEqual(make_judge.call_args.args[0], "target-key")

    def assert_live_failure_aborts(self, query_fn, judge_fn):
        args = self.live_args()
        with self.one_probe_file(), patch.dict(os.environ, {
            "OPENCODE_GO_API_KEY": "target-key",
            "OPENROUTER_API_KEY": "judge-key",
        }, clear=True), patch.object(
            ikp_v2.v1, "make_query_fn", return_value=query_fn
        ), patch.object(
            ikp_v2.v1, "make_judge_fn", return_value=judge_fn
        ), patch.object(
            ikp_v2, "robust_estimate"
        ) as estimate, patch.object(ikp_v2, "show") as show, contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                ikp_v2.run_live(args)

        self.assertEqual(raised.exception.code, 1)
        estimate.assert_not_called()
        show.assert_not_called()

    def test_target_api_failure_aborts_without_an_estimate(self):
        query_fn = Mock(side_effect=ikp_v2.v1.TargetAPIError("target unavailable"))
        judge_fn = Mock()

        self.assert_live_failure_aborts(query_fn, judge_fn)
        judge_fn.assert_not_called()

    def test_judge_api_failure_aborts_without_an_estimate(self):
        query_fn = Mock(return_value="Paris")
        judge_fn = Mock(side_effect=ikp_v2.v1.JudgeAPIError("judge unavailable"))

        self.assert_live_failure_aborts(query_fn, judge_fn)
        query_fn.assert_called_once()
        judge_fn.assert_called_once()


class IKPEstimateV2ArtifactTests(unittest.TestCase):
    def run_from_results(self, artifact):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(json.dumps(artifact))
            args = SimpleNamespace(from_results=str(path), split="all")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                with self.assertRaises(SystemExit) as raised:
                    ikp_v2.run_from_results(args)
            return raised.exception.code

    def test_failed_v1_artifact_is_rejected(self):
        self.assertEqual(self.run_from_results({
            "status": "failed",
            "results": [{
                "tier": "T1",
                "verdict": "INFRASTRUCTURE_ERROR",
            }],
        }), 1)

    def test_infrastructure_and_unknown_verdicts_are_rejected(self):
        for verdict in ("INFRASTRUCTURE_ERROR", "MAYBE"):
            with self.subTest(verdict=verdict):
                self.assertEqual(self.run_from_results({
                    "status": "completed",
                    "results": [{"tier": "T1", "verdict": verdict}],
                }), 1)

    def test_missing_or_unknown_tier_and_required_fields_are_rejected(self):
        records = [
            {"verdict": "CORRECT"},
            {"tier": "T8", "verdict": "CORRECT"},
            {"tier": "T1"},
            "not a record",
        ]
        for record in records:
            with self.subTest(record=record):
                self.assertEqual(self.run_from_results({
                    "status": "completed",
                    "results": [record],
                }), 1)

    def test_legacy_valid_artifact_without_status_is_rescored(self):
        artifact = {
            "model": "legacy-model",
            "results": [{"tier": "T1", "verdict": "CORRECT"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(artifact))
            args = SimpleNamespace(from_results=str(path), split="all")
            with patch.object(ikp_v2, "show") as show:
                ikp_v2.run_from_results(args)

        show.assert_called_once()
        self.assertEqual(show.call_args.args[0], "legacy-model")
        self.assertEqual(show.call_args.args[2], 1)


if __name__ == "__main__":
    unittest.main()
