"""Send a coding brief to the local model and write the returned code block to a file.

  python tools/delegate.py <brief.md> <output file> [--feedback <file>] [--name <label>]

Each attempt is logged under results/delegations/<UTC timestamp>-<label>/ (brief, raw reply, stats).
"""
import argparse
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8080/v1/chat/completions"
SYSTEM = ("You are a senior C engineer. Follow the brief exactly. Reply with ONE fenced code block "
          "containing the complete file, and nothing else.")
CODE_BLOCK = re.compile(r"```[a-zA-Z0-9]*\n(.*?)```", re.DOTALL)


def ask(prompt: str) -> tuple[str, dict, float]:
    body = {"model": "default_model", "max_tokens": 8192, "temperature": 0.2,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        payload = json.load(resp)
    return payload["choices"][0]["message"]["content"] or "", payload.get("usage", {}), time.perf_counter() - start


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("brief"); p.add_argument("output")
    p.add_argument("--feedback", help="file with previous code + test output to fix")
    p.add_argument("--name", default="task")
    a = p.parse_args()

    prompt = Path(a.brief).read_text()
    if a.feedback:
        prompt += "\n\n## Your previous attempt failed\n\n" + Path(a.feedback).read_text() + \
                  "\n\nFix it. Reply with the complete corrected file."
    reply, usage, secs = ask(prompt)

    log = ROOT / "results" / "delegations" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{a.name}"
    log.mkdir(parents=True)
    (log / "prompt.md").write_text(prompt)
    (log / "reply.md").write_text(reply)
    stats = {"seconds": round(secs, 1), "usage": usage, "output": a.output}
    (log / "stats.json").write_text(json.dumps(stats, indent=2))

    match = CODE_BLOCK.search(reply)
    if not match:
        raise SystemExit(f"no code block in reply, see {log / 'reply.md'}")
    Path(a.output).write_text(match.group(1))
    print(f"wrote {a.output} in {secs:.1f}s ({usage.get('completion_tokens')} tokens) - log {log}")


if __name__ == "__main__":
    main()
