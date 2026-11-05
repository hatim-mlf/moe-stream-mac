# Night 1 pipeline check (run in the daytime, 2026-11-05, after a reboot, apps closed, wired limit 28672 MB)

Memory finding: the planned config (max_seq_length 10,240) does NOT fit on the 32 GB M1 Max.
- probe1 (4 rows ~500 tok + one 5,269-tok row): steps 1-4 ran, OOM (Metal kIOGPU...OutOfMemory) on the 5,269 step.
  Forward-only validation at 9,153 tokens ran fine. 8 GB swap during the run.
- probe2 (rows 1,200-2,968 tok): OOM during training. probe3 (470, 471, 1,200, 2,016): OOM on the first step.
- probe4 (470, 471 tok only): OK, MLX peak 25.14 GiB, ~12 s/step.
=> ~25 GiB at ~500 tokens with ~28 GiB usable: headroom ~3 GiB, i.e. well under ~1k tokens.
Likely cause: mlx_lm default_loss builds full-vocab (248,320) logits over every position, fp32 CE + grad.
Fix in progress: completion-only chunked loss (brief 32). Until then: rows <= 600 tokens only.
Night 1 training set: data/finetune/night1 (22/31 train rows, 6/10 valid rows <= 600 tok), 66 iters.
- train1 (4 layers, rows <= 600): OOM at step 12 (rows 0.5-0.6k). probeA (1 layer, 470 tok): MLX peak 19.48 GiB vs
  25.14 at 4 layers => ~1.9 GiB per LoRA layer (weights on disk 19 GB). Layer count, not length, dominates at short
  rows (plausibly quantized-expert dequant in the MoE backward). train2: 2 layers, rows <= 600, 66 iters.
- train2 (2 layers, 22 rows <= 600 tok, 66 iters, lr 1e-5, scale 20): OK, ~4 min, MLX peak 24.46 GiB,
  val loss 0.763 -> 0.337 -> 0.277 -> 0.249 (steps 1/22/44/66), train loss ~0.01-0.06 at the end (overfit).
- eval: adapter answers short prompts correctly and stops (48 tokens), but on heldout-v2 it collapses into a
  repetition loop (`#include` lines repeated to the 8,192-token cap, sample-output.txt): 2/2 tasks failed at
  ~2.8 min each. Full 30-task eval not run (would be ~80 min of capped loops). Verdict: pipeline works end to end;
  this adapter is a NO-GO (22 tiny rows, 3 epochs, overfit).
- probeB (2 layers, rows 1,200-2,968 tok): OOM on the first step. Length cost dominates above ~1k tokens even at
  2 layers => completion-only chunked loss (brief 32) is required before training on real rows.
- probeC (1 layer, rows 1,200-2,968 tok): OK, MLX peak 24.19 GiB (vs 19.48 at 470 tok) => ~1.9 GiB per extra 1k tokens at 1 layer. 1-layer LoRA on rows <= 3k tokens (28/31 rows) is the config that fits today.
