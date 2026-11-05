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
