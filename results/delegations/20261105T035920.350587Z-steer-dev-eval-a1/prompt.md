# Task: write tools/steer_dev_eval.py (Python 3.14, standard library only, plus `import ft_eval` from tools/)

Measure how often the model, served by the local MLX server, solves the steering DEV tasks.

Data: data/steering/pairs.jsonl, one JSON object per line with keys pair_id, task_id, category, split ("build" or
"dev"), messages (a list of {"role","content"}: system then user), preferred, unpreferred. Each task has a test at
data/steering/tests/<task_id>/test.py (Python) or test.c (C). The language of a pair is "python" if its test.py
exists, otherwise "c".

Write exactly:

```python
ROOT = Path(__file__).resolve().parents[1]

def load_pairs(split: str) -> list[dict]:
    """Rows of data/steering/pairs.jsonl whose "split" equals split, in file order. ValueError if none."""

def run_candidate(task_id: str, code: str) -> tuple[bool, str]:
    """In a fresh tempfile.TemporaryDirectory: write code + "\n" to candidate.py or candidate.c.
    Python: run [sys.executable, <test.py>] with cwd=temp and PYTHONPATH=temp prepended to the environment's
    PYTHONPATH. C: compile [cc, "-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic", candidate.c, <test.c>, "-o",
    temp/"check"] (cc = shutil.which("cc")); if compilation fails return (False, "compile failed:\n" + output); else
    run the binary. Every subprocess uses text=True, stdout=PIPE, stderr=STDOUT, timeout=60 (a timeout returns
    (False, "timeout")). Return (returncode == 0, output)."""

def ask_server(endpoint: str, messages: list) -> str:
    """POST {"model": "default_model", "temperature": 0, "max_tokens": 4096,
    "chat_template_kwargs": {"enable_thinking": False}, "messages": messages} as JSON to endpoint with
    urllib.request (timeout 1800); return choices[0].message.content or ""."""

def evaluate(pairs: list[dict], ask) -> list[dict]:
    """For each pair: reply = ask(pair["messages"]); code = ft_eval.extract_code(reply); passed, output =
    run_candidate(task_id, code) if code else (False, "no code block"). Return rows {"task_id", "category",
    "passed", "output_tail": output[-2000:]}."""

def main(argv=None) -> int:
    """argparse: --split (default "dev"), --endpoint (default ft_eval.API), --label (required), --out (required
    path). rows = evaluate(load_pairs(split), lambda m: ask_server(endpoint, m)). Write {"label", "split",
    "passed": count, "total": len(rows), "rows": rows} as JSON (indent 1) to --out and print
    "<label>: <passed>/<total>". Return 0."""
```
Import ft_eval with `sys.path.insert(0, str(ROOT / "tools"))` then `import ft_eval`. End with
`if __name__ == "__main__": raise SystemExit(main())`.
Rules: each function under 30 lines; no other functions. Return only the complete file in one fenced python block.
