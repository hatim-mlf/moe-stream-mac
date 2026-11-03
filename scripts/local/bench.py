"""Benchmark a local OpenAI-compatible server (mlx_lm.server or llama-server): prefill and decode speed.

Usage:
  python scripts/local/bench.py --base-url http://127.0.0.1:8080/v1 --name mlx-qwen38-27b-4bit
Writes results/<UTC timestamp>-<name>/metrics.json. Standard library only.
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Filler that is roughly one token per word for Qwen tokenizers; the server's usage field gives the real count.
FILLER_WORD = "data "
PROMPT_WORDS = (500, 2000, 8000, 16000)
QUESTION = "\n\nIgnore the text above. Write a 200-word explanation of how a hash map works."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--name", required=True, help="label: <engine>-<model>-<quant>")
    parser.add_argument("--model", default="default", help="model field sent to the server")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sizes", type=int, nargs="*", default=list(PROMPT_WORDS))
    return parser.parse_args()


def stream_once(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first_token_at = None
    chunks = 0
    usage = None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[len("data:"):])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    chunks += 1
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
    end = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("server streamed no tokens")

    prompt_tokens = (usage or {}).get("prompt_tokens")
    completion_tokens = (usage or {}).get("completion_tokens") or chunks
    ttft = first_token_at - start
    decode_s = end - first_token_at
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "usage_reported": usage is not None,
        "ttft_s": round(ttft, 3),
        "prefill_tok_s": round(prompt_tokens / ttft, 1) if prompt_tokens else None,
        "decode_tok_s": round((completion_tokens - 1) / decode_s, 2) if decode_s > 0 else None,
    }


def median(rows: list, key: str):
    values = [row[key] for row in rows if row[key] is not None]
    return statistics.median(values) if values else None


def mac_info() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version()}
    for key in ("hw.memsize", "machdep.cpu.brand_string", "iogpu.wired_limit_mb"):
        try:
            info[key] = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
        except OSError:
            info[key] = None
    return info


def main() -> None:
    args = parse_args()
    print("warm-up...")
    stream_once(args.base_url, args.model, "Say OK.", 8)

    results = []
    for words in args.sizes:
        # Vary the prefix per repeat so a server-side prompt cache can't make prefill look free.
        runs = []
        for repeat in range(args.repeats):
            prompt = f"[run {repeat}] " + FILLER_WORD * words + QUESTION
            runs.append(stream_once(args.base_url, args.model, prompt, args.max_tokens))
        summary = {
            "prompt_words": words,
            "prompt_tokens": runs[0]["prompt_tokens"],
            "median_ttft_s": median(runs, "ttft_s"),
            "median_prefill_tok_s": median(runs, "prefill_tok_s"),
            "median_decode_tok_s": median(runs, "decode_tok_s"),
            "runs": runs,
        }
        results.append(summary)
        print(f"{words:>6} words | prompt {summary['prompt_tokens']} tok | "
              f"prefill {summary['median_prefill_tok_s']} tok/s | decode {summary['median_decode_tok_s']} tok/s")

    out_dir = ROOT / "results" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{args.name}"
    out_dir.mkdir(parents=True)
    report = {
        "name": args.name, "base_url": args.base_url, "model": args.model,
        "max_tokens": args.max_tokens, "repeats": args.repeats, "temperature": 0,
        "machine": mac_info(), "results": results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
