"""Ask the local MLX model from the shell. Standard library only.

  python scripts/local/ask.py "Explain this error" < log.txt
  git diff | python scripts/local/ask.py "Write a commit message for this diff"

Piped stdin is appended to the prompt. Streams the answer to stdout and stats to stderr.
Env: LOCAL_LLM_URL (default http://127.0.0.1:8080/v1).
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:8080/v1")
MAX_TOKENS = 4096


def build_prompt(argv: list[str]) -> str:
    instruction = " ".join(argv).strip()
    piped = "" if sys.stdin.isatty() else sys.stdin.read()
    if not instruction and not piped:
        sys.exit('usage: ask.py "instruction" [< input]')
    return f"{instruction}\n\n{piped}".strip()


def stream_answer(prompt: str) -> None:
    body = {
        "model": "default_model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first_at, usage = None, None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[len("data:"):])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if text:
                    first_at = first_at or time.perf_counter()
                    sys.stdout.write(text)
                    sys.stdout.flush()
    end = time.perf_counter()
    sys.stdout.write("\n")
    if usage and first_at:
        decode_rate = usage["completion_tokens"] / max(end - first_at, 1e-6)
        sys.stderr.write(
            f"[local] {usage['prompt_tokens']} in / {usage['completion_tokens']} out tokens, "
            f"first token {first_at - start:.2f}s, {decode_rate:.1f} tok/s\n"
        )


def main() -> None:
    try:
        stream_answer(build_prompt(sys.argv[1:]))
    except urllib.error.URLError as error:
        sys.exit(f"local model unreachable at {BASE_URL} ({error.reason}). Start it: bash scripts/local/serve-mlx.sh")


if __name__ == "__main__":
    main()
