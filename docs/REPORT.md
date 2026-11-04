# Report: running a 122B model locally on a 32 GB Mac (3–4 Nov 2026)

**Machine:** MacBook Pro M1 Max, 10-core CPU (8 performance + 2 efficiency), 32-core GPU, 32 GB RAM, ~1 TB SSD.
**Real-world condition:** Xcode, browsers, Codex, Claude Code sessions and a background job stay open
(~15 GB of other apps, swap often 25–30 GB). All numbers below are measured that way unless stated.

## 1. What we set out to do

Run a big open model that doesn't fit in RAM, locally, and use a local model as a helper for Claude.
Started from the pure-C engine in [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c), which runs a
2.78T MoE model by keeping the small "trunk" in RAM and streaming the experts it needs from the SSD.

## 2. Timeline

| Step | What | Result |
|---|---|---|
| C2C paper | Planned to verify Cache-to-Cache | Dropped: needs open models in one process and a trained fuser per pair, so it can't apply to Claude/Codex |
| Upstream build | Built kimi-k3-in-c, ran its tests | Passes on the M1 Max (needs `brew install libomp`) |
| Local helper (Track A) | Qwen3.6-35B-A3B 4-bit on MLX | 49 tok/s replies, ~660 tok/s prompt reading |
| Claude Code on Qwen | LiteLLM translator → MLX | Works, but ~2.5 min per turn (Claude Code's huge system prompt). Translator itself costs ~30 ms |
| Shell helper | `scripts/local/ask.py` | 6 s for a 4,200-token file; best way to use the local model |
| **C engine port** | Ported the C engine to Qwen 3.5/3.6 MoE (MLX 4/8-bit) | Whole model matches MLX float32 to ~1e-6, 24/24 tokens |
| 35B in C | streaming experts from SSD | 1.39 GB RAM (MLX: ~19 GB) |
| **122B 8-bit in C** | 131 GB model, 4× RAM | **Runs, 5.5 GB RAM, matches MLX float32 to ~2e-6**; MLX can't load it at all |
| Speed work (CPU) | threads → NEON → parallel expert reads | 122B: 8.1 → 0.67 s/token (12×) |
| Terminal chat | `engine/cli/qchat.py` | multi-turn, streaming, thinking, /threads, stats footer |
| **Hybrid GPU engine** | MLX GPU for the math + our expert streaming | Long prompts 9× faster on the 122B; output identical to plain MLX |
| GPU expert cache | keep hot experts in GPU memory | **Net loss**: steals RAM from the macOS file cache |
| Next-layer prediction | guess layer L+1's experts during layer L | **92–94% accurate**, but fresh buffers per layer cause memory churn: slower for now, so off by default |

## 3. Final numbers

### Correctness

| Check | Result |
|---|---|
| C engine vs MLX float32, 35B and 122B | 24/24 positions same top token, logits within 1–3e-6 |
| C engine, 240-token prompt (35B) | 255/255 positions match MLX float32 |
| Chat-template path (35B) | identical to MLX |
| Hybrid GPU vs plain MLX (bf16) | bit-identical, 255/255 |
| Note | MLX's default bf16 differs from float32 by 1–8% on logits (6/255 top tokens); the C engine is the more exact one |

### Speed and memory (8 threads unless noted; chat defaults to 6 threads, nice 10)

| Model | Engine | Prompt reading | Reply | RAM |
|---|---|---|---|---|
| 35B 4-bit (20 GB) | MLX (GPU, whole model) | ~660 tok/s | ~49 tok/s | ~19 GB |
| 35B 4-bit | C engine | ~4 tok/s | 6.2 tok/s | 1.4 GB |
| 35B 4-bit | hybrid GPU | 10–18 tok/s | 6.3–7.4 tok/s | ~2.5 GB |
| 122B 8-bit (131 GB) | MLX | can't load | — | — |
| 122B 8-bit | C engine | 1.2 tok/s | 1.5 tok/s (0.67 s/tok) | 5.5–6.7 GB |
| 122B 8-bit | hybrid GPU | **10.4 tok/s** (240-tok prompt) | 1.24 tok/s | ~2–8 GB + GPU |

122B reply speed history: 8.12 s/token (1 thread) → 3.69 (4 threads) → 3.14 (8) → 1.21 (NEON) → 0.67 (parallel reads).

### Why the 122B is ~5× slower than the 35B

Per generated token the 122B needs ~3.8 GB of experts (35B: ~0.5 GB): experts 2× wider, model 1.5× wider,
8-bit vs 4-bit, 48 vs 40 layers. The 35B's experts mostly stay in RAM; the 122B's 123 GB never fit, so they come
from the SSD. Realistic floor on this Mac: ~0.6 s/token for any engine.

### Real-world finding (1,000-word story on the 122B, Xcode etc. open)

Started at 1.26 tok/s and degraded to ~0.3 tok/s over an hour: macOS compressed/swapped the model's own memory
(the chat process kept 0.3 GB resident, 0.5 GB free, 13 GB in the compressor, 28 GB swap).

## 4. What didn't work, and why

| Idea | Why it failed |
|---|---|
| Claude Code / Codex driving the local model | Their system prompts are huge; the local model re-reads them every turn (~2.5 min) |
| 10 CPU threads | Slower than 8: the efficiency cores hold the others back |
| GPU expert cache (6–10 GB) | GPU memory and macOS file cache are the same RAM; 0.81 → 5.5 s/token |
| Next-layer prediction (as built) | 92% accurate, but allocating fresh buffers per layer churns ~10 GB/token and macOS compresses the GPU trunk (3 s/token) |
| MLX in float32 for the reference | Exceeds the GPU wired-memory limit on 32 GB; solved by loading one layer at a time |

## 5. Working with the local Qwen (35B 4-bit) as a helper

16 delegated tasks (`docs/delegation-log.md`):
- **First-try pass:** 3 of 15 Qwen tasks (MoE block, parallel fetch, disk-order reader), each with a skeleton.
- **Average attempts:** ~2.3. **Lead fixes needed:** 7 of 15.
- **Good at:** focused math, filling a skeleton with fixed signatures, commit messages (wrote the last 9).
- **Weak at:** restructuring code, juggling buffers, following review notes (ignored the same instruction twice
  on several tasks), and it repeated the same signed-value bug in two tasks.
- **Rules that emerged:** the lead writes specs, tests and skeletons; review every diff for memory safety;
  never send Qwen text containing its own control tokens (`<|im_end|>`, `<think>`); stop its 20 GB server
  before any 122B run.

## 6. How to use what exists

```bash
cd ~/moe-stream-mac
.venv/bin/python engine/cli/qchat.py --model ~/models/Qwen3.6-35B-A3B-4bit             # fast, 2.5 GB
.venv/bin/python engine/cli/qchat.py --model ~/models/Qwen3.5-122B-A10B-8bit           # GPU hybrid
.venv/bin/python engine/cli/qchat.py --model ~/models/Qwen3.5-122B-A10B-8bit --engine cpu  # C engine, f32
python3 scripts/local/ask.py "Summarise this" < file.txt                                # shell helper (needs serve-mlx.sh)
```

## 7. Open questions

1. **Which local model can code well enough to help without constant correction?** The 35B needs too much
   supervision; the 122B is better but slow here.
2. **Memory under real use:** reusable read buffers + reads that bypass the file cache, so the engine stops
   fighting the user's apps (and prediction can be re-enabled).
3. **4-bit 122B:** half the data per token (~2× faster, half the RAM); quality cost vs 8-bit not yet measured here.
