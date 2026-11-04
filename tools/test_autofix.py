"""Offline tests for tools.autofix."""
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import autofix


class AutofixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.brief = self.root / "brief.md"
        self.brief.write_text("Implement f(x).", encoding="utf-8")
        self.logs = self.root / "logs"
        self.root_patch = patch.object(autofix, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def command_for(self, output: Path, expression: str) -> str:
        snippet = ("import sys; sys.dont_write_bytecode = True; sys.path.insert(0, "
                   + repr(str(self.root)) + "); "
                   "import mod; " + expression)
        return shlex.quote(sys.executable) + " -c " + shlex.quote(snippet)

    def run_cli(self, output: Path, command: str, fake, *extra: str) -> tuple[int, list[str]]:
        prompts = []

        def ask(prompt, system):
            prompts.append(prompt)
            return fake[len(prompts) - 1], {"completion_tokens": 4}, 0.01

        with patch.object(autofix.delegate, "ask", side_effect=ask):
            code = autofix.main([str(self.brief), str(output), "--test", command,
                                 "--name", "unit", *extra])
        return code, prompts

    def test_failure_feedback_then_pass(self):
        output = self.root / "mod.py"
        command = self.command_for(output, "assert mod.f(2) == 4")
        replies = ["```python\ndef f(x):\n    return x + 1\n```",
                   "```python\ndef f(x):\n    return x * 2\n```"]
        code, prompts = self.run_cli(output, command, replies)
        self.assertEqual(code, 0)
        self.assertEqual(len(prompts), 2)
        self.assertIn("AssertionError", prompts[1])
        self.assertEqual(output.read_text(encoding="utf-8"), "def f(x):\n    return x * 2\n")

    def test_all_attempts_fail_restores_original(self):
        output = self.root / "mod.py"
        original = "def f(x):\n    return 9\n"
        output.write_text(original, encoding="utf-8")
        command = self.command_for(output, "assert mod.f(2) == 4")
        replies = ["```python\ndef f(x):\n    return 1\n```",
                   "```python\ndef f(x):\n    return 3\n```"]
        code, _ = self.run_cli(output, command, replies, "--max-attempts", "2")
        self.assertEqual(code, 1)
        self.assertEqual(output.read_text(encoding="utf-8"), original)

    def test_functions_replace_preserves_control_token_string(self):
        output = self.root / "mod.py"
        original = 'def f(x):\n    return x + 1\n\nTOKEN = "<|im_end|>"\n'
        output.write_text(original, encoding="utf-8")
        command = self.command_for(output, "assert mod.f(2) == 4")
        replies = ["```python\ndef f(x):\n    return x * 2\n```"]
        code, _ = self.run_cli(output, command, replies, "--functions", "f")
        self.assertEqual(code, 0)
        self.assertEqual(output.read_text(encoding="utf-8"),
                         'def f(x):\n    return x * 2\n\nTOKEN = "<|im_end|>"\n')


if __name__ == "__main__":
    result = unittest.main(exit=False)
    if result.result.wasSuccessful():
        print("All tests passed.")
    else:
        raise SystemExit(1)
