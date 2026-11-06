# Models

Weights are never committed (CLAUDE.md). This is the list of every model the project has used, with its
official source, so any of them can be restored with `scripts/fetch-model.sh <repo> <dir>` (resumable, verifies
each file's SHA-256 against the Hub). The Mac has room for about one or two big models at a time; delete the
ones not in use and re-fetch them when needed.

| Local dir (`~/models/…`) | Hugging Face repo | Original weights | Size | Used for | On disk (2026-11-06) |
|---|---|---|---|---|---|
| `Qwen3.6-35B-A3B-4bit` | `mlx-community/Qwen3.6-35B-A3B-4bit` | `Qwen/Qwen3.6-35B-A3B` | 19 GB | local helper (`scripts/local/serve-mlx.sh`), fine-tuning and eval base | yes |
| `Qwen3.5-122B-A10B-8bit` | `mlx-community/Qwen3.5-122B-A10B-8bit` | `Qwen/Qwen3.5-122B-A10B` | 122 GB | main target of the SSD-streaming engines (`engine/`, results/20261104T121244Z-engine-122b-8bit-v1) | no (deleted 2026-11-06) |
| `Qwen3-Coder-Next-4bit` | `mlx-community/Qwen3-Coder-Next-4bit` | `Qwen/Qwen3-Coder-Next` | — | coding-model research, merge experiments (Phase 1b) | yes |
| `KAT-Coder-V2.5-Dev-OptiQ-4bit` | `mlx-community/KAT-Coder-V2.5-Dev-OptiQ-4bit` | `Kwaipilot/KAT-Coder-V2.5-Dev` | — | task-vector merge experiments (Phase 1b) | yes |
| `merge-kat-lam0.5` | — (built locally by `tools/merge_task_vector.py`) | Qwen3.6-35B-A3B + KAT task vector | — | merge experiment output | yes |
| `Qwen3.8-27B-official-mlx-4bit` | `mlx-community/Qwen3.8-27B-4bit` | `Qwen/Qwen3.8-27B` | 16.1 GB | dense 27B comparison (official weights) | yes |
| `Qwen3.8-27B-TFCF-NM-DAU-mlx-6bit` | `Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-mlx-6Bit` | `Qwen/Qwen3.8-27B` → DavidAU uncensored fine-tune | 22.7 GB | community fine-tune comparison: ties Qwen3.6 on heldout-v2, 5× slower | yes |

Notes:
- Qwen does not publish MLX builds itself; `mlx-community` conversions of the official weights are the source.
- Downloads: Hugging Face drops long single connections on this network. `aria2c` with 8 parallel ranged
  connections reached 30-40 MB/s where `hf download` and plain `curl` stalled (2026-11-06).
- After a reboot, re-run `sudo sysctl iogpu.wired_limit_mb=31744` before running a model.
