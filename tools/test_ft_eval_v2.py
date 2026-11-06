#!/usr/bin/env python3
"""Unit tests for heldout-v2 frozen evaluation metadata and pairing."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from unittest import mock
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
                ft_eval.paired_report(self._pair(Path(folder), base, other), allow_method_diff=True)

    def test_paired_report_rejects_sample_count_mismatch_even_when_methods_may_differ(self):
        with tempfile.TemporaryDirectory() as folder:
            base = self._payload()
            other = self._payload()
            other["samples"] = 3
            with self.assertRaisesRegex(ValueError, "sample count"):
                ft_eval.paired_report(self._pair(Path(folder), base, other), allow_method_diff=True)

    def test_paired_report_rejects_method_mismatch_unless_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            base = self._payload()
            other = self._payload()
            other["guide_sha256"] = "guide-hash"
            files = self._pair(Path(folder), base, other)
            with self.assertRaisesRegex(ValueError, "guide"):
                ft_eval.paired_report(files)
            with contextlib.redirect_stdout(io.StringIO()):
                ft_eval.paired_report(files, allow_method_diff=True)
            other["guide_sha256"] = None
            other["plan_first"] = True
            files = self._pair(Path(folder), base, other)
            with self.assertRaisesRegex(ValueError, "plan-first"):
                ft_eval.paired_report(files)
            with contextlib.redirect_stdout(io.StringIO()):
                ft_eval.paired_report(files, allow_method_diff=True)

    def test_default_request_body_matches_legacy_baseline(self):
        bodies = []

        def fake_urlopen(request, timeout):
            bodies.append(request.data)
            response = {"choices": [{"message": {"content": "ok"}}]}
            return io.BytesIO(json.dumps(response).encode())

        prompt = "captured baseline prompt"
        expected_body = {
            "model": "local", "temperature": 0.0, "max_tokens": 8192,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": ft_eval.SYSTEM},
                         {"role": "user", "content": prompt}],
        }
        with mock.patch.object(ft_eval.urllib.request, "urlopen", fake_urlopen):
            ft_eval.request_model(prompt, "http://localhost/test", "local", None)
        self.assertEqual(bodies, [json.dumps(expected_body).encode()])

    def test_default_result_metadata_matches_legacy_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result.json"
            args = ft_eval.arguments(["--split", "public", "--out", str(output)])
            ft_eval.save_results([], args)
            expected = {
                "benchmark_revision": ft_eval.FROZEN_REVISION, "split": "public", "suite": "public",
                "model": "default_model", "adapter": "base", "attempt_cap": 1,
                "endpoint": ft_eval.API, "temperature": 0.0, "seed": None, "samples": 1,
                "max_tokens": 8192, "review_notes": False, "results": [],
            }
            self.assertEqual(output.read_text(), json.dumps(expected, indent=2) + "\n")

    def test_guide_is_used_for_initial_and_repair_requests_and_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            guide = directory / "guide.md"
            guide.write_text("Use checked boundaries.\n")
            output = directory / "result.json"
            args = ft_eval.arguments(["--split", "public", "--guide", str(guide),
                                      "--attempts", "2", "--out", str(output)])
            task = {"id": "HumanEval/test", "entry": "candidate", "prompt": "SPEC",
                    "tests": "assert True\n"}
            calls = []

            def fake_request(prompt, endpoint, model, adapter, temperature, seed, **kwargs):
                calls.append((prompt, kwargs))
                return "```python\nanswer\n```"

            with mock.patch.object(ft_eval, "request_model", side_effect=fake_request), \
                    mock.patch.object(ft_eval, "install_and_check", side_effect=[(1, "failure"), (0, "ok")]) as check:
                result = ft_eval.evaluate_one(task, args)
            ft_eval.save_results([result], args)

            expected_system = ft_eval.SYSTEM.rstrip("\n") + "\n\nUse checked boundaries.\n"
            self.assertEqual(len(calls), 2)
            self.assertEqual([call[1]["system_prompt"] for call in calls], [expected_system] * 2)
            self.assertIn("## Attempt 1 failed\n\nYour code:", calls[1][0])
            self.assertEqual(check.call_count, 2)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["guide"], str(guide))
            self.assertEqual(payload["guide_sha256"], ft_eval.hashlib.sha256(guide.read_bytes()).hexdigest())
            self.assertNotIn("plan_first", payload)

    def test_plan_first_orders_plan_and_code_calls_and_saves_plans(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result.json"
            args = ft_eval.arguments(["--split", "public", "--plan-first", "--attempts", "2",
                                      "--out", str(output)])
            task = {"id": "HumanEval/test", "entry": "candidate", "prompt": "SPEC",
                    "tests": "assert True\n"}
            first_plan = "Plan line\n" + ("x" * 2100)
            second_plan = "Review the failing test, then fix the boundary."
            replies = iter([first_plan, "```python\nfirst solution\n```", second_plan,
                            "```python\nrepaired solution\n```"])
            calls = []
            tested = []

            def fake_request(prompt, endpoint, model, adapter, temperature, seed, **kwargs):
                calls.append((prompt, kwargs))
                return next(replies)

            def fake_check(task_id, filename, code, public_test=None):
                tested.append(code)
                return (1, "failed assertion") if len(tested) == 1 else (0, "ok")

            with mock.patch.object(ft_eval, "request_model", side_effect=fake_request), \
                    mock.patch.object(ft_eval, "install_and_check", side_effect=fake_check):
                result = ft_eval.evaluate_one(task, args)
            ft_eval.save_results([result], args)

            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[0][0], "SPEC\n\n" + ft_eval.PLAN_INSTRUCTION)
            self.assertNotIn("messages", calls[0][1])
            first_history = calls[1][1]["messages"]
            self.assertEqual(first_history, [
                {"role": "user", "content": "SPEC"},
                {"role": "assistant", "content": first_plan},
                {"role": "user", "content": ft_eval.CODE_INSTRUCTION},
            ])
            self.assertIn("## Attempt 1 failed\n\nYour code:", calls[2][0])
            self.assertIn(ft_eval.PLAN_INSTRUCTION, calls[2][0])
            second_history = calls[3][1]["messages"]
            self.assertIn("## Attempt 1 failed\n\nYour code:", second_history[0]["content"])
            self.assertEqual(second_history[1], {"role": "assistant", "content": second_plan})
            self.assertEqual(tested, ["first solution", "repaired solution"])
            self.assertEqual(result["plans"], [first_plan[:2000], second_plan])
            self.assertTrue(all(len(plan) <= 2000 for plan in result["plans"]))
            payload = json.loads(output.read_text())
            self.assertIs(payload["plan_first"], True)
            self.assertEqual(payload["results"][0]["plans"], result["plans"])

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
