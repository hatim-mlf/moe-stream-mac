# c2c-eval

Independent verification of **Cache-to-Cache (C2C)** ([paper](https://arxiv.org/abs/2510.03215), [code](https://github.com/thu-nics/C2C), ICLR'26), and a go/no-go decision on whether it can improve our multi-agent working environment.

C2C lets two LLMs exchange information by projecting and fusing one model's **KV-cache** into another's, so they skip generating and re-reading intermediate text.

## Claims to verify

| # | Claim (from paper / upstream README) | Baseline |
|---|---|---|
| C1 | +8.5–10.5% accuracy | receiver model working alone |
| C2 | +3.0–5.0% accuracy | text-to-text (T2T) communication between the same two models |
| C3 | ~2× lower latency (upstream README says 2.0×; some summaries say 2.5×) | T2T communication |

## The fit question, answered first

C2C needs **open-weight models running in the same process** (raw KV-cache access) and a **trained fuser for each model pair**. It cannot be applied to API-only models such as Claude, GPT and Codex, because they don't expose their KV-cache. So for our current agent stack (Claude Code, Codex and other API agents talking through text and the shell), C2C **cannot be a drop-in replacement**.

It becomes relevant only if we run local open models (Qwen, Llama and so on, via vLLM or transformers) as cheap sub-agents, for example a small "worker" model fed context by a larger "sharer". Phase 3 of the [plan](docs/plan.md) evaluates that.

## Layout

```
docs/plan.md           phased verification plan + go/no-go criteria
scripts/setup.sh       clones upstream C2C at a pinned commit, installs deps
scripts/smoke_test.py  receiver-alone vs C2C on a few prompts, with timing
results/               one folder per run (metrics + env info committed, raw outputs ignored)
third_party/C2C        upstream code (gitignored, created by setup.sh)
```

## Quick start

```bash
bash scripts/setup.sh                      # creates .venv, clones upstream at pinned SHA
source .venv/bin/activate
python scripts/smoke_test.py --device cpu  # Qwen3-0.6B receiver + Qwen2.5-0.5B sharer, works on CPU
```

For the real benchmarks (Phase 2) you need a CUDA GPU. See [docs/plan.md](docs/plan.md).
