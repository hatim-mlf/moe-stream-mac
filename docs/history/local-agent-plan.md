# Local worker model under Claude Code: plan

Goal: run an open-weight model locally on an **M1 Max (32 GB unified memory, 32-core GPU)** at
**15–20 tok/s or better**, and let Claude Code delegate heavy, well-scoped work to it (bulk
summarising, first drafts, boilerplate, classification). Claude stays the orchestrator and
checks every local output before using it.

This is Track A. Track B (`plan.md`) ports the kimi-k3-in-c engine so that 8-bit MoE models
bigger than RAM can stream from SSD. Track A's benchmark is the baseline that Track B has to beat.

## The arithmetic that decides everything

Decode speed is capped by memory bandwidth divided by the bytes of weights read per token.
For the M1 Max, assume about 400 GB/s peak and roughly 250–300 GB/s in practice (Phase 0
measures it). macOS lets the GPU wire about 21–24 GB by default. Raising it with
`sudo sysctl iogpu.wired_limit_mb=<MB>` resets on reboot and has to leave room for macOS
and Claude Code.

| Model | Type | Bits | Weights | Fits? | Decode ceiling (at ~280 GB/s) |
|---|---|---|---|---|---|
| Qwen3.8-27B | dense | 8 | ~29 GB | **no** (all 32 GB, nothing left for the OS) | ~10 tok/s even if it fitted |
| Qwen3.8-27B | dense | 6 | ~22 GB | tight; needs a raised wired limit | ~13 tok/s |
| Qwen3.8-27B | dense | 4–5 | ~15–18 GB | yes | ~16–19 tok/s |
| Qwen3.6-35B-A3B | MoE, ~3B active | 4–6 | 20–29 GB | yes / tight | ~60+ tok/s (reads only ~2–3 GB per token) |
| Qwen3.6-35B-A3B | MoE | 8 | ~38 GB | **no** (Track B) | n/a |
| Qwen3-Coder-30B-A3B | MoE, code-tuned, older | 4–6 | ~17–25 GB | yes / tight | ~60+ tok/s |

Real numbers come in below these ceilings. So a **dense model at 8-bit is out on this
machine**, and the 15–20 tok/s target with a dense 27B needs 4–5-bit weights. Disk-streaming
engines (the kimi-k3-in-c approach) don't help a dense model, because every token touches
every weight.

The question Phase 1 answers is therefore: **Qwen3.8-27B at 4–6 bits vs Qwen3.6-35B-A3B (with Qwen3-Coder-30B-A3B as a code-tuned alternative)
at 4–6 bits. Which gives the better quality per second on our real tasks?**

## Phases

### Phase 0: measure the machine (10 min)
- Memory bandwidth: run `mlx` or a STREAM build and record the GB/s actually achieved.
- Record `sysctl hw.memsize machdep.cpu.brand_string` and `sysctl iogpu.wired_limit_mb`.

### Phase 1: speed and quality, MLX vs llama.cpp
Serve each candidate with both engines (one at a time, so they don't compete for memory):

```bash
mlx_lm.server --model <mlx-community repo> --port 8080
llama-server -hf <gguf repo>:<quant> --jinja -c 32768 --port 8080
```

Run `python scripts/local/bench.py --base-url http://127.0.0.1:8080/v1 --name <engine>-<model>-<quant>`.
It records, for each prompt size (about 0.5k, 2k, 8k and 16k tokens):
- **prefill tok/s** (time to first token). This is the real bottleneck for agent work, where
  Claude hands over thousands of tokens of files.
- **decode tok/s**.

Quality, per quant:
- Compare against a higher-precision reference with the KL divergence over a fixed text
  (`llama-perplexity --kl-divergence-base` / `--kl-divergence`). The 8-bit reference may have
  to run from SSD via mmap, which is slow but only needs to happen once.
- Task check: 30–50 real delegation tasks (`tasks/`, to be written). Claude grades each output
  pass/fail blind to which model produced it.

Pass: a candidate at **≥15 tok/s decode**, **≥300 tok/s prefill at 8k**, and a task pass rate
the operator considers worth delegating.

### Phase 2: the Claude Code bridge
`scripts/local/mcp_bridge.py` is an MCP server that exposes the local model to Claude Code as
tools (`local_generate`, `local_process_files`). Register it with:

```bash
claude mcp add local-llm -- python ~/kimi-k3-c-engine/scripts/local/mcp_bridge.py
```

Measure over two weeks of real use:
- how many tasks were delegated,
- how many local outputs Claude rejected or rewrote,
- the Claude usage saved (estimated from the tokens the local model produced).

### Phase 3: only if Phase 1–2 leave a gap
- Speculative decoding with a small Qwen draft model (bigger win on dense 27B than on MoE).
- Prompt-cache reuse across delegations that share a prefix (the same files sent twice).
- A custom engine only if a measured bottleneck remains that MLX and llama.cpp can't close.

## Go/no-go
- **Go**: one candidate meets the Phase 1 thresholds, and over Phase 2 Claude rejects under
  ~25% of delegated outputs.
- **No-go**: nothing reaches 15 tok/s at usable quality, or rejections make delegation slower
  than doing the task in Claude directly.
