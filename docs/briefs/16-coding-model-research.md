<goal>
Find the open-weight AI coding models (released or updated in 2026) that a local coding assistant on THIS machine can
actually rely on, and rank them with evidence. Write the result to docs/research/coding-models-2026-11.md in this
repo. This is research only: do not download any model file larger than 1 GB, do not edit any other file, do not commit.
</goal>

<why>
We run local models next to Claude as coding helpers. The current one (Qwen3.6-35B-A3B, 4-bit, ~3B active params)
passed only 3 of 15 delegated coding tasks on the first try (see docs/delegation-log.md): it drifts from specs,
ignores review notes and mishandles buffers. We need a model that writes correct code from a precise spec with
fewer corrections, and runs fast enough on this Mac.
</why>

<constraints_of_this_machine>
- MacBook Pro M1 Max, 32 GB unified memory, 32-core GPU, internal SSD (~6.5 GB/s sequential reads measured).
- Real workload stays open (Xcode, browsers, other agents): only ~12-16 GB of RAM can be given to a model.
- Free disk: ~100 GB (one large model at a time).
- Our engines (see README.md, docs/C_ENGINE.md, docs/MLX_ENGINE.md) run MLX affine-quantized checkpoints.
  `qwen3_5_moe` works in both engines today; other MoE architectures run through MLX + our expert streaming only
  if mlx-lm supports them; dense models must fit in RAM and run with plain MLX.
- Speed model (use it, show your arithmetic): per generated token an MoE model reads its ACTIVE expert weights:
  bytes/token ~= layers x top_k x 3 x hidden x moe_intermediate x bits/8 (+ shared experts and attention from RAM).
  Expert data that doesn't fit in ~10 GB of file cache comes from the SSD. Measured reference points on this Mac:
  Qwen3.6-35B-A3B 4-bit ~0.5 GB/token -> 6.2-7.4 tok/s; Qwen3.5-122B-A10B 8-bit ~3.8 GB/token -> 1.2-1.5 tok/s.
  A dense model reads ALL its weights per token from RAM (~150-200 GB/s usable): tok/s ~= bandwidth / weight bytes,
  only if the weights fit in the 12-16 GB budget.
</constraints_of_this_machine>

<how_to_work>
Work like a senior ML engineer doing model selection, not like a blogger:
1. Start from RAW sources, not "top 10" articles or aggregator rankings:
   - Hugging Face Hub API JSON (e.g. https://huggingface.co/api/models?search=coder&sort=lastModified&direction=-1,
     https://huggingface.co/api/models/<id>?expand[]=safetensors&expand[]=cardData&expand[]=createdAt,
     https://huggingface.co/<id>/resolve/main/config.json): exact architecture, total/active params, layers, experts,
     top_k, hidden, moe_intermediate_size, license, creation date. Also check mlx-community for MLX conversions and
     their sizes per bit width.
   - Official technical reports / release posts (arXiv, vendor blogs) for training cutoff and claimed scores.
   - Independent leaderboards with raw data: SWE-bench Verified / SWE-bench Pro, LiveCodeBench (note the date window
     vs the model's training cutoff = contamination check), Aider polyglot (leaderboard data in the aider repo),
     Terminal-Bench. Prefer results produced by the leaderboard maintainers over vendor self-reports; label which is which.
2. Use the camofox MCP browser for ALL web access (navigate_and_snapshot works for JSON API pages too); the shell
   sandbox has no network.
3. Cross-check every claimed score against at least one independent source; flag disagreements.
4. Compute bytes/token and predicted tok/s for 4-, 6- and 8-bit with the speed model above; say whether it fits the
   RAM budget (dense) or streams (MoE), and which of our engines can run it today.
5. Rank by expected COST PER ACCEPTED CODING TASK on this Mac (quality from independent coding benchmarks x speed),
   not by raw benchmark score.
</how_to_work>

<deliverable>
docs/research/coding-models-2026-11.md containing:
1. Selection criteria (one short list).
2. Candidate table (at least 10 models considered): model id, release date, architecture (model_type), total /
   active params, license, MLX availability + size per bit width, bytes/token and predicted tok/s, engine
   compatibility, independent coding scores (with source + whether the benchmark window is after the training
   cutoff). EVERY number links to its source URL.
3. Shortlist of exactly 3 models to test locally, with the download plan that fits ~100 GB.
4. Rejected models with one-line reasons.
5. Open questions / data you could not verify.
6. How we will verify: the shortlist gets replayed on our 15 delegated coding tasks (docs/briefs/01-15 with their
   automatic tests in engine/tests and engine/hybrid/test_*.py); first-try pass rate, attempts, and time per task
   decide, not the leaderboards.
</deliverable>

<report>
Reply with: the shortlist (3 models) with one line each, the 3 numbers you are least sure about, and the file path.
</report>
