"""MCP server that lets Claude Code delegate work to a local model behind an OpenAI-compatible API.

Register:  claude mcp add local-llm -- python /abs/path/scripts/local/mcp_bridge.py
Env:       LOCAL_LLM_URL (default http://127.0.0.1:8080/v1), LOCAL_LLM_MODEL (default "default"),
           LOCAL_LLM_LOG (default results/delegations.jsonl, gitignored)
Requires:  pip install mcp
"""

import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("LOCAL_LLM_MODEL", "default")
LOG_PATH = Path(os.environ.get("LOCAL_LLM_LOG", ROOT / "results" / "delegations.jsonl"))
MAX_FILE_BYTES = 200_000

mcp = FastMCP("local-llm")


def _chat(prompt: str, system: str, max_tokens: int, tool: str) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        payload = json.load(response)
    seconds = time.perf_counter() - start
    text = payload["choices"][0]["message"].get("content") or ""

    # Delegation log: what Phase 2 measures (volume, speed, tokens Claude didn't have to produce).
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as log:
        log.write(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(), "tool": tool, "model": MODEL,
            "seconds": round(seconds, 2), "usage": payload.get("usage"),
        }) + "\n")
    return text


@mcp.tool()
def local_generate(prompt: str, system: str = "", max_tokens: int = 2048) -> str:
    """Run a self-contained task on the local model and return its raw output.

    Use for bulk or mechanical work: drafts, boilerplate, rewrites, classification.
    The local model is weaker than Claude: check the output before using it.
    """
    return _chat(prompt, system, max_tokens, "local_generate")


@mcp.tool()
def local_process_files(paths: list[str], instruction: str, max_tokens: int = 2048) -> str:
    """Read the given files locally and apply one instruction to them (e.g. summarise, list TODOs).

    The file contents go to the local model only, so they don't consume Claude context.
    Check the output before relying on it.
    """
    parts = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        data = path.read_bytes()[:MAX_FILE_BYTES].decode(errors="replace")
        parts.append(f"=== {path} ===\n{data}")
    prompt = f"{instruction}\n\n" + "\n\n".join(parts)
    return _chat(prompt, "", max_tokens, "local_process_files")


if __name__ == "__main__":
    mcp.run()
