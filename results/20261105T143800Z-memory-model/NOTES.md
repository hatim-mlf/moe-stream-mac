# Memory model run notes (brief 36)

Isolated real-weight block benchmarks (`mem_profile.py block`, value_and_grad of sum(block(x)) w.r.t. x,
incremental peak above the 18.17 GiB of loaded weights; raw: blocks*.jsonl):

| block (layer) | T | forward | value_and_grad |
|---|---|---|---|
| MoE (39) | 1000 / 2000 | 0.12 / 0.24 | 0.19 / 0.63 GiB |
| attention (39) | 1000 / 2000 | 0.08 / 0.22 | 0.30 / 0.48 GiB |
| GDN (38), ckpt off | 128 / 256 / 512 / 1000 | 0.41 / 0.60 / 1.17 / — | 1.14 / 2.19 / 4.29 / mem_guard stop (> 11) GiB |
| GDN (38), ckpt on | 128 / 256 / 512 | 0.35 / 0.50 / 0.52 | 0.89 / 1.62 / 3.68 GiB |

- GDN backward ~8.2 MiB/token without checkpoint, ~7.2 with: ~4 recurrent states (32x128x128 fp32 = 2 MiB)
  alive per token. The chunked mx.checkpoint saves only ~15%; whole-graph lazy evaluation probably keeps every
  chunk's recompute alive (open question; check in the GPU capture).
- GDN forward alone keeps ~1 state per token (~2.3 MiB/token).
- Extrapolated: one GDN LoRA layer at 3k tokens needs ~21-24 GiB for its backward → cannot fit; at 600 tokens
  ~4.5 GiB. 1-layer LoRA (layer 39, attention) fits at 3k because no GDN layer is in the backward path.
- Open: in the full model, the probeI/J A/B showed no checkpoint effect (22.532 vs 22.528), the block test
  shows ~15%.

Isolated step vs mlx_lm trainer, T=1000 (compare.jsonl, cachelimit-*.jsonl; peaks are MLX active):

| L | isolated `step` | `trainer-step` (default cache) | trainer, cache limit 0 | trainer, cache limit 2 GiB |
|---|---|---|---|---|
| 1 | 19.29 | 19.27 | — | — |
| 2 | 23.21 (cache 6.55 at end) | mem_guard stop 29.59 wired | **26.36, fits** (25.6 s incl. load) | mem_guard stop 29.88 wired |

- mem_guard / the OS count MLX's buffer cache (freed buffers kept for reuse; default limit ~1.5 x 26.8 GB);
  `mx.get_peak_memory` does not. System wired ~ active + cache + ~1.7 GiB OS. The budget must model the cache.
- Open: trainer active peak at L=2 is 3.15 GiB above the isolated step (26.36 vs 23.21). Candidates: trainer
  default GDN_CHECKPOINT, padding to 32, mask from the real prompt offset, step/optimizer in one graph. Not traced.

Held-out prediction points (fixed 15:25, before any data on them): trainer-step, cache limit 0,
(a) L=3 T=768, (b) L=2 T=1500. Predictions come from the revised mem_budget.py, written here before the runs.
Sweep started 15:25: trainer-step, cache limit 0, T {256,512,1000,2000,3000} x L {1,2,3,4} (sweep.jsonl).

Sweep result (15:45; trainer-step, cache limit 0; MLX active peak GiB / max system wired GiB):

| T | L=1 | L=2 | L=3 | L=4 |
|---|---|---|---|---|
| 256 | 18.60 / 21.32 | 20.22 / 23.20 | 20.25 / 23.99 | 20.25 / 23.84 |
| 512 | 18.87 / 21.29 | 22.34 / 24.59 | 22.39 / 25.51 | 22.32 / 24.51 |
| 1000 | 19.39 / 21.95 | 26.36 / 28.92 | guard stop | 26.45 / 29.03 |
| 2000 | 19.85 / 22.04 | guard stop | guard stop | guard stop |
| 3000 | 20.11 / 22.52 | guard stop | guard stop | guard stop |

- The requested form peak = W + aL + bT + cLT fits badly (R² 0.72): peak does NOT grow with L beyond 2. With
  grad_checkpoint only one layer's backward is live at a time, so peak = max over LoRA layers, not a sum.
- Empirical law (lead, from the table): active ≈ 18.45 GiB + 0.55 MiB·T + [any GDN layer in LoRA set] · 6.95 MiB·T.
  GDN term from L≥2 minus L=1: 1.62 / 3.47 / 6.97 GiB at T = 256 / 512 / 1000 (6.3 / 6.9 / 7.1 MiB/token),
  matching the isolated GDN block with checkpoint (~7.2 MiB/token). System wired ≈ active + 2.3-3.5 GiB.
- Practical limit at the 29.5 GiB guard: any L ≥ 2 fits up to ~900-1000 tokens (L=3/L=4 at T=1000 sit at the
  limit: one stop, one pass at 29.03); L=1 fits at 3k with ~7 GiB to spare.

Lead predictions from this law, written BEFORE running (held-out points):
- (a) L=3 T=768:  active 18.45 + 0.41 + 5.21 = 24.07 GiB; system wired ~26.5-27.5 → fits.
- (b) L=2 T=1500: active 18.45 + 0.81 + 10.18 = 29.44 GiB; system wired > 31 → mem_guard stop.
C18's revised budget predictions for (a)/(b) to be added when it returns (it has not seen the sweep).
C18 revised budget (source-derived, GDN = 4 state-sized arrays per token = 8 MiB/token; has not seen the sweep),
recorded before reading the held-out results:
- (a) L=3 T=768:  active 24.59 GiB, system wired 26.29 → fits.
- (b) L=2 T=1500: active 30.68 GiB, system wired 32.38 → mem_guard stop.
Held-out results (16:00): (a) L=3 T=768: active 24.46 GiB, max wired 27.18 → fits. Lead law −1.6%, C18 budget
+0.5% (wired −3.3%). (b) L=2 T=1500: mem_guard stop at 29.52 GiB wired, as both predicted. Both within 5%.

Deep LoRA (trainer-step, cache limit 0; active / max wired GiB / wall s incl. ~7 s load):
- L=8  T=900: 25.66 / 28.58 / 30.0
- L=16 T=900: 25.70 / 29.27 / 50.1   (0.23 GiB under the guard: too tight to use)
- L=16 T=768: 24.40 / 28.36 / 40.9
Active peak stays flat in L (one GDN backward live at a time). Wired − active grows with L (2.9 → 3.6 GiB from
L=8 to 16): not in MLX's counters; open question (Metal residency / command buffers?). Compute time grows with L.

Open question #1 (why gdn_checkpoint does not help), diagnostics in repro/ (16:30):
- ckpt_toy.py: 2 MiB-state recurrence, gradient w.r.t. the initial state only: mx.checkpoint WORKS (backward
  flat ~0.37-0.43 GiB for T = 256-1024; plain grows 0.76 → 2.35 GiB).
- gdn_iso.py: our gdn_checkpoint vs gated_delta_ops alone, real GDN shapes, grads w.r.t. q,k,v,g,beta:
  ops 9.6 / 8.9 MiB/token, ckpt 7.9 / 6.0 MiB/token at T = 256 / 512 (grad w.r.t. g only: 2.9 MiB/token).
- ckpt_toy2.py: same toy, but gradients w.r.t. the per-step inputs q,k,v (as training needs): checkpoint memory
  GROWS with T again: 1.03 / 1.58 / 2.69 GiB at T = 256 / 512 / 1024 (~1 state/token; plain ~3.3 states/token).
=> Reproduced outside our code: MLX 0.32.4 mx.checkpoint does not bound backward memory of a chunked recurrence
   once the chunk inputs need cotangents. Mechanism not yet traced (candidates: evaluation order of the chained
   per-chunk vjps keeps state-sized temporaries alive; broadcast-multiply vjp materialises state-sized cotangents
   per step). Candidate upstream MLX issue with ckpt_toy2.py as the minimal repro.
