#!/usr/bin/env python3
"""Unit tests for heldout-v2 frozen evaluation metadata and pairing."""
import contextlib
import importlib.util
import io
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ft_eval_v2_under_test", ROOT / "tools" / "ft_eval.py")
ft_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ft_eval)


class EvalV2Tests(unittest.TestCase):
    def _pair(self, directory, left, right):
        first = directory / "first.json"
        second = directory / "second.json"
        first.write_text(json.dumps(left))
        second.write_text(json.dumps(right))
        return first, second

    def _payload(self):
        return {"suite": "heldout-v2", "suite_sha256": "hash-a", "benchmark_revision": "hash-a",
                "split": "heldout-v2", "attempt_cap": 1, "review_notes": False, "temperature": 0,
                "samples": 2, "model": "local", "results": [
                    {"task_id": "case-a", "first_pass": 0.5, "passed": 1.0, "attempt_score": 1.0}]}

    def test_paired_report_rejects_temperature_and_suite_mismatch(self):
        with self.subTest("temperature"), __import__("tempfile").TemporaryDirectory() as folder:
            base = self._payload()
            other = self._payload()
            other["temperature"] = 0.2
            with self.assertRaisesRegex(ValueError, "temperature"):
                ft_eval.paired_report(self._pair(Path(folder), base, other))
        with self.subTest("suite"), __import__("tempfile").TemporaryDirectory() as folder:
            base = self._payload()
            other = self._payload()
            other["suite"] = "other-suite"
            with self.assertRaisesRegex(ValueError, "suite"):
                ft_eval.paired_report(self._pair(Path(folder), base, other))

    def test_sample_aggregate_uses_per_sample_means(self):
        rows = [
            {"task_id": "case-a", "first_pass": True, "passed": True, "attempts": 1,
             "attempt_score": 1, "output_tail": "ok"},
            {"task_id": "case-a", "first_pass": False, "passed": False, "attempts": 2,
             "attempt_score": 3, "output_tail": "fail"},
        ]
        aggregate = ft_eval.aggregate_samples("case-a", rows)
        self.assertEqual(aggregate["first_pass"], 0.5)
        self.assertEqual(aggregate["passed"], 0.5)
        self.assertEqual(aggregate["attempts"], 1.5)
        self.assertEqual(aggregate["attempt_score"], 2.0)
        self.assertEqual(len(aggregate["samples"]), 2)

    def test_manifest_hash_mismatch_is_refused(self):
        manifest = json.loads((ROOT / "tools" / "ft_eval_heldout_v2.json").read_text())
        manifest["tasks"][0]["sha256"]["spec.md"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "manifest mismatch"):
            ft_eval.verify_v2_manifest(manifest=manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
