# Local helper tools (optional)

Tools used during the experiment to run a local model next to Claude Code. Not needed for the engines.

```bash
bash scripts/local/serve-mlx.sh                               # MLX OpenAI-compatible server on 127.0.0.1:8080
python3 scripts/local/ask.py "Explain this error" < log.txt   # shell helper: fastest way to use the local model
open scripts/local/chat.html                                  # small browser chat with tok/s readout
python3 scripts/local/bench.py --name mlx-35b                 # prefill/decode benchmark
bash scripts/local/claude-local.sh                            # Claude Code on the local model via LiteLLM (slow: ~2.5 min/turn)
```

Findings: Claude Code and Codex send very large system prompts, so a local model spends minutes re-reading them
every turn; a plain shell command (`ask.py`) with a short prompt answers in seconds. See docs/REPORT.md.

`tools/delegate.py` and `tools/qwen_commit.py` show how the local Qwen was used to write code and commit messages
under Claude's review; `docs/delegation-log.md` records every attempt.
