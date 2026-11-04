# Task: write tools/route_prompts.py (Python 3.14, numpy + mlx_lm only)

Tokenize a JSON list of prompts into the C engine's prompt format, using the model's chat template, so we can record
routing traces under the same conditions as the evaluation.

Input JSON: a list of objects {"id": str, "prompt": str}. Output: one file <out_dir>/<id>.i32 per prompt, holding
the token ids as little-endian int32 (np.array(ids, "<i4").tofile(path)), nothing else.

Write exactly:

```python
def chat_ids(tokenizer, system: str | None, prompt: str) -> list[int]:
    """messages = [{"role": "system", "content": system}] if system else [], then the user message.
    Return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
    enable_thinking=False) as a list of int."""

def load_prompts(path: Path) -> list[dict]:
    """Read the JSON list; raise ValueError if it is not a list, if an item lacks a str "id" or str "prompt",
    if an id is not made of [A-Za-z0-9_-] only, or if ids repeat."""

def main(argv=None) -> int:
    """argparse: --model DIR (required), --prompts FILE (required), --out DIR (required, created if missing),
    --system-file FILE (optional; its text is the system message). Load the tokenizer with
    mlx_lm.utils.load_tokenizer(Path(model)). For each prompt write <out>/<id>.i32 and print "<id> <n> tokens".
    Return 0."""
```
End with `if __name__ == "__main__": raise SystemExit(main())`.
Rules: each function under 30 lines; no other functions. Return only the complete file in one fenced python block.
