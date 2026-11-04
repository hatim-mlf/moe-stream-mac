"""Commit the staged changes with a message written by the local Qwen model.

  python tools/qwen_commit.py [--all] [--note "results / context for the message"] [--dry-run]

Qwen sees the staged diff (truncated) and the note, and writes a conventional commit message. The tool
validates the format, appends the attribution trailer and runs `git commit`. Nothing is pushed.
"""
import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8080/v1/chat/completions"
MAX_DIFF_CHARS = 14000
TYPES = ("feat", "fix", "refactor", "docs", "test", "chore", "perf", "ci")
SUBJECT = re.compile(rf"^({'|'.join(TYPES)}): \S.{{0,70}}$")
TRAILER = ("Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01WmSoJ2A8P527GB6DM4hLdY")
PROMPT = """Write a git commit message for the staged changes below.

Format, exactly:
<type>: <imperative summary, max 72 chars>

- <bullet: what changed and why, with key numbers from the note if relevant>
- <2 to 5 bullets total, each under 100 chars>

<type> is one of: {types}. No other text, no code fences, no sign-off.

Note from the lead engineer:
{note}

git diff --cached --stat:
{stat}

git diff --cached (may be truncated):
{diff}
"""


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def defang(text: str) -> str:
    """Escape the model's own control tokens so they cannot end or reroute its reply."""
    return text.replace("<|", "<\\|").replace("<think>", "<think_>").replace("</think>", "</think_>")


def ask_qwen(prompt: str) -> str:
    body = {"model": "default_model", "max_tokens": 400, "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return (json.load(resp)["choices"][0]["message"]["content"] or "").strip()
    except OSError as err:
        raise SystemExit(f"Qwen server unreachable ({err}). Start it: bash scripts/local/serve-mlx.sh")


def clean(message: str) -> str:
    """Strip fences/whitespace and force git's layout: subject, one blank line, body."""
    lines = [l.rstrip() for l in message.strip().strip("`").strip().splitlines()]
    body = [l for l in lines[1:] if l.strip()]
    return "\n".join([lines[0], "", *body]).strip() if lines else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="stage all changes first (git add -A)")
    ap.add_argument("--note", default="(none)", help="context or results Qwen should mention")
    ap.add_argument("--dry-run", action="store_true", help="print the message, do not commit")
    a = ap.parse_args()
    if a.all:
        git("add", "-A")
    stat = git("diff", "--cached", "--stat")
    if not stat.strip():
        raise SystemExit("nothing staged")
    diff = git("diff", "--cached")
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + f"\n... [truncated, {len(diff) - MAX_DIFF_CHARS} more chars]"
    prompt = PROMPT.format(types=", ".join(TYPES), note=defang(a.note), stat=defang(stat), diff=defang(diff))
    for attempt in (1, 2):
        message = clean(ask_qwen(prompt))
        lines = message.splitlines()
        if lines and SUBJECT.match(lines[0]) and any(l.startswith("- ") for l in lines[1:]):
            break
        print(f"attempt {attempt}: message rejected (subject format or no '- ' bullets): {lines[:1]}", file=sys.stderr)
        prompt += ("\n\nYour previous answer was rejected. Start with '<type>: <summary>', then a blank line, "
                   "then 2 to 5 lines starting with '- '.")
    else:
        raise SystemExit("Qwen did not produce a valid commit message; nothing committed")
    full = f"{message}\n\n{TRAILER}\n"
    print(full)
    if a.dry_run:
        return
    msg_file = Path(git("rev-parse", "--git-dir").strip()) / "QWEN_COMMIT_MSG"
    msg_file.write_text(full)
    subprocess.run(["git", "commit", "-q", "-F", str(msg_file)], check=True)
    print(git("log", "--oneline", "-1").strip())


if __name__ == "__main__":
    main()
