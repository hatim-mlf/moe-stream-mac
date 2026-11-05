# Task: write tools/route_nll.py (Python 3.14, numpy + mlx + mlx_lm only)

Measure how likely the model finds known-good answers (teacher-forced negative log-likelihood), with or without a
router bias from tools/router_bias.py. Used as the development metric for the router-bias sweep.

Input: a JSONL file; each line {"task_id": str, "messages": [..., {"role": "assistant", "content": str}]} where the
LAST message is the assistant answer to score.

Write exactly:

```python
def split_ids(tokenizer, messages: list) -> tuple[list[int], list[int]]:
    """prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True,
    enable_thinking=False); full = tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=False).
    Raise ValueError if the last message is not role "assistant" or full[:len(prompt)] != prompt.
    Return (prompt, full) as lists of int."""

def completion_nll(model, prompt: list[int], full: list[int]) -> tuple[float, int]:
    """logits = model(mx.array([full[:-1]])) -> shape [1, len(full)-1, vocab]. Convert to float32 and take
    log-softmax over the vocab (logits - mx.logsumexp(logits, axis=-1, keepdims=True)). The answer tokens are
    full[len(prompt):]; token full[j] is predicted at position j-1. Return (sum of -log p over the answer tokens as a
    Python float, number of answer tokens)."""

def score_file(model, tokenizer, path: Path) -> dict:
    """For every line: (prompt, full) = split_ids(...); nll, n = completion_nll(...). Return
    {"rows": [{"task_id", "nll", "tokens", "nll_per_token"}...], "total_nll", "total_tokens",
    "nll_per_token": total_nll / total_tokens}."""

def main(argv=None) -> int:
    """argparse: --model DIR (required), --data JSONL (required, repeatable: action="append"), --bias NPY (optional),
    --mode (choices router_bias.MODES, default "select"), --scale FLOAT (default 1.0; the bias array is multiplied by
    it), --out JSON (required). model, tokenizer = mlx_lm.load(model). If --bias: router_bias.install_router_bias(
    model, np.load(bias) * scale, mode). Write {"model", "bias", "mode", "scale", "files": {data path: score_file
    result}} to --out (indent=1) and print one line per file: "<path> nll/token=<value:.4f> tokens=<n>". Return 0."""
```
Import router_bias with `sys.path.insert(0, str(Path(__file__).resolve().parent))` then `import router_bias`.
End with `if __name__ == "__main__": raise SystemExit(main())`.
Rules: each function under 30 lines; no other functions. Return only the complete file in one fenced python block.
