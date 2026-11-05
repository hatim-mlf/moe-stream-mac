# Frozen evaluation protocol

## Frozen assets and split

The repository suite is task-disjoint from the natural fine-tuning examples. `tools/ft_eval_manifest.json`
pins the benchmark source revision (`6f43cc8712a6a7a99d325e1db09b0377cd48a6d3`), the SHA-256 of each first-attempt
prompt and review prompt, and the SHA-256 of the exact system prompt (including `tools/helper_rules.md`). The
evaluator checks these hashes before use. It creates a temporary tree from that Git revision, writes the
candidate into the target file, and runs the fixture tests without modifying the working tree.

| Split | Task IDs | Use |
|---|---|---|
| Train | `gdn-layer`, `attn-layer`, `moe-layer`, `qchat`, `fetch-experts`, `read-sorted` | Natural SFT and preference/review examples may use these tasks. |
| Validation | `model-forward`, `expert-reader` | Natural examples are validation-only; never put these tasks in training or preference files. |
| Held out | `qmat`, `ops`, `gdn`, `qmat-row`, `qgroup`, `expert-cache`, `predictive`, `pinned` | Never used in SFT, preference, review-following, teacher prompts, or train/valid data. Use for final paired project evaluation. |
| Public mini | HumanEval IDs 5, 7, and 58 | Separate public regression check; never used in dataset generation. |

“Train eligible” therefore means the first two rows together (8 tasks), not that validation examples may be
trained on. The split was chosen at task level, with related code changes grouped under the same task. Eight
held-out tasks are a small pilot set; they support paired case-level evidence, not a broad claim about all coding.

Held-out task tests are offline and frozen by the benchmark source revision:

| Task | Gate |
|---|---|
| `qmat`, `qmat-row` | Compile `build/test_qmat`; run all tracked qmat fixtures. |
| `ops`, `gdn`, `qgroup` | Compile and run `test_ops`, `test_gdn`, and `test_qgroup` against tracked fixtures (1,800 qgroup cases). |
| `expert-cache` | `engine/hybrid/test_expert_cache.py` |
| `predictive` | Frozen fake-reader fixture embedded in `tools/ft_eval.py`; checks hits, misses, row shapes, and pending cleanup. |
| `pinned` | `engine/hybrid/test_pinned.py` |

The public mini uses the HumanEval task IDs `intersperse`, `filter_by_substring`, and `common`, with frozen local
edge-case assertions in `tools/ft_eval.py`. It runs offline once the local MLX server is available, but it is not
the official EvalPlus pass@1 score or the full HumanEval+ test suite. The Qwen model card does not state a
training-data cutoff, so I cannot certify these old public tasks as post-cutoff or contamination-free. They are
a small forgetting/regression check only. [Qwen3.6 model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B),
[EvalPlus project](https://github.com/evalplus/evalplus).

## Heldout-v2: frozen local coding evaluation

`heldout-v2` is a new, independently frozen 30-task suite for measuring the local Qwen3.6-35B-A3B 4-bit model before and after weight edits or fine-tuning. It uses the unchanged `tools/helper_rules.md` system prompt. The model receives one `spec.md` per task, or one `review.md` for the 10 review-note tasks; the generic runner writes the response into a temporary task copy and runs its frozen build and test commands. `tools/ft_eval_heldout_v2.json` pins every prompt, task contract, and test asset and records the suite hash.

Each `evals/heldout-v2/<id>/` contains `spec.md`, `task.json`, `tests/`, `reference/`, `mutants/`, and, for exactly 10 tasks, `review.md`. Fixtures are optional and none are needed in this suite. The mix is 12 C99 kernels, 12 Python/NumPy utilities, and 6 skeleton-fill tasks (3 C and 3 Python); difficulty is balanced 10/10/10.

The revised tests map every behavioral spec clause to named cases in the suite coverage table. They contain 3,648
test-program lines (templates and coverage documentation excluded), with 67 minimum and 121 median lines per
task. Each task has 240 or 256 fixed-seed randomized oracle cases, including a larger stress input, and four
mutants that cover distinct errors; the suite has 120 mutants total. The slowest reference test took 0.823 seconds
in this workspace. One spec clarification sets the packed-rank
exception in `p_affine_groups` to `TypeError`, matching the frozen reference's existing behavior. See
[`tests/COVERAGE.md`](../../evals/heldout-v2/c_axis_record/tests/COVERAGE.md) for the requirement-to-case map.

| ID | Language | Kind | Difficulty | Diagnosis classes |
|---|---|---|---|---|
| `c_axis_record` | c | kernel | easy | Spec drift, invented behavior, or omitted requested content, API, return-value, or data-shape misuse |
| `c_bits3_decode` | c | kernel | medium | Buffer bounds, offsets, or aliasing, Quantized signedness or group indexing |
| `c_echo_ring` | c | kernel | medium | Buffer bounds, offsets, or aliasing, API, return-value, or data-shape misuse |
| `c_group_lse` | c | kernel | hard | Buffer bounds, offsets, or aliasing, Required behavior omitted |
| `c_rank_topk` | c | kernel | medium | Buffer bounds, offsets, or aliasing, Required behavior omitted |
| `c_rms_gate` | c | kernel | hard | API, return-value, or data-shape misuse, Spec drift, invented behavior, or omitted requested content |
| `c_sat8_pair` | c | kernel | easy | Buffer bounds, offsets, or aliasing, Required behavior omitted |
| `c_scan_checked` | c | kernel | medium | Buffer bounds, offsets, or aliasing, API, return-value, or data-shape misuse |
| `c_skeleton_clamp` | c | skeleton | easy | Buffer bounds, offsets, or aliasing, Ignored or partly applied review note |
| `c_skeleton_pair` | c | skeleton | medium | API, return-value, or data-shape misuse, Ignored or partly applied review note |
| `c_skeleton_window` | c | skeleton | hard | Buffer bounds, offsets, or aliasing, Ignored or partly applied review note |
| `c_span_alias` | c | kernel | hard | Buffer bounds, offsets, or aliasing, Required behavior omitted |
| `c_stable_norm` | c | kernel | medium | Buffer bounds, offsets, or aliasing, Required behavior omitted |
| `c_strided_proj` | c | kernel | medium | API, return-value, or data-shape misuse, Buffer bounds, offsets, or aliasing |
| `c_u4_groupdot` | c | kernel | hard | Quantized signedness or group indexing, Spec drift, invented behavior, or omitted requested content |
| `p_affine_groups` | python | utility | hard | Quantized signedness or group indexing, API, return-value, or data-shape misuse |
| `p_axis_layout` | python | utility | easy | API, return-value, or data-shape misuse, Buffer bounds, offsets, or aliasing |
| `p_axis_take` | python | utility | medium | API, return-value, or data-shape misuse, Required behavior omitted |
| `p_block_header` | python | utility | easy | API, return-value, or data-shape misuse, Spec drift, invented behavior, or omitted requested content |
| `p_bytes_lru` | python | utility | hard | API, return-value, or data-shape misuse, Buffer bounds, offsets, or aliasing |
| `p_chunk_plan` | python | utility | easy | Buffer bounds, offsets, or aliasing, Required behavior omitted |
| `p_dtype_blend` | python | utility | easy | API, return-value, or data-shape misuse, Spec drift, invented behavior, or omitted requested content |
| `p_lex_order` | python | utility | hard | API, return-value, or data-shape misuse, Required behavior omitted |
| `p_mask_out` | python | utility | medium | Buffer bounds, offsets, or aliasing, Spec drift, invented behavior, or omitted requested content |
| `p_runs_merge` | python | utility | hard | API, return-value, or data-shape misuse, Required behavior omitted |
| `p_segment_pool` | python | utility | medium | API, return-value, or data-shape misuse, Required behavior omitted |
| `p_skeleton_calibrate` | python | skeleton | hard | Required behavior omitted, Ignored or partly applied review note |
| `p_skeleton_delta` | python | skeleton | easy | Quantized signedness or group indexing, Ignored or partly applied review note |
| `p_skeleton_diags` | python | skeleton | easy | API, return-value, or data-shape misuse, Ignored or partly applied review note |
| `p_window_copy` | python | utility | easy | Buffer bounds, offsets, or aliasing, Required behavior omitted |

Freeze and run the offline gates:

```bash
.venv/bin/python tools/ft_heldout_v2_freeze.py
.venv/bin/python tools/ft_eval.py --split heldout-v2 --self-check
.venv/bin/python tools/ft_eval.py --split heldout-v2 --mutant-check
.venv/bin/python tools/ft_leak_check.py
```

Run the local model for a paired measurement, including optional seeded repeated samples:

```bash
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --split heldout-v2 --attempts 1 --temperature 0 --samples 1 --out results/finetune/base-heldout-v2.json
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --split heldout-v2 --attempts 1 --temperature 0 --samples 1 --out results/finetune/edited-heldout-v2.json
.venv/bin/python tools/ft_eval.py --paired results/finetune/base-heldout-v2.json results/finetune/edited-heldout-v2.json
```

Use `--review-notes` to measure the frozen review prompts; it accepts one response per task. `--samples K` sends K separate requests per task and stores the per-sample rows plus their task-level means. `--seed N` adds a deterministic request seed (each sample uses `N + sample_index`). Paired reports require identical suites and hashes, task IDs, retry caps, review mode, model, temperature, and sample count.

**Never-train rule:** no heldout-v2 spec, review prompt, model response, reference solution, test output, task ID, or public function name may be used in SFT, preference, review-following, teacher prompts, or train/validation data. `ft_build_dataset.py` excludes both held-out manifests and rejects heldout-v2 public names; `ft_leak_check.py` scans `data/finetune/` and other `evals/` files for leaks.

## Metrics

- **First-attempt pass rate:** fraction of tasks whose first response has one fenced code block and passes the
  entire offline suite. With `--review-notes`, this is the one-shot review-note suite pass rate.
- **Pass at feedback cap:** fraction passing after at most `--attempts N` responses. On failure the evaluator
  sends the previous code and exact test output back to the model, then reruns the complete suite.
- **Attempts to pass:** mean response count among solved tasks. Also report `censored_attempt_score`, the mean
  of attempts for solved tasks and `N+1` for unsolved tasks, so a model cannot look better by failing hard cases.
- **Review-note acceptance:** fraction of the eight held-out retry prompts whose response passes the full
  corresponding suite once. This is a test-backed proxy for compliance; it does not score “change nothing else”
  or prose-only notes unless a suite enforces them. Record any manual minimal-diff review separately.
- **Paired change:** on identical task IDs, compute adapter-minus-base first-pass and pass-at-cap differences;
  compute attempts-score in the lower-is-better direction. `--paired BASE.json ADAPTER.json` reports a
  deterministic 10,000-resample paired bootstrap percentile 95% interval over tasks (seed `20261104`).

Run inference with temperature 0 and the frozen system prompt. Save the JSON output from every run; pair only
files with the same split, retry cap, review-note mode, base model, and exact task IDs.

## Run the baseline

Terminal A starts exactly one MLX model server from the local 4-bit weights:

```bash
MODEL=/Users/hatim/models/Qwen3.6-35B-A3B-4bit scripts/local/serve-mlx.sh
```

Terminal B runs inference and the fixture harness. This first baseline uses one response per task:

```bash
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --split heldout --attempts 1 --out results/finetune/base-first.json
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --split heldout --attempts 3 --out results/finetune/base-feedback.json
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --split heldout --review-notes --out results/finetune/base-review.json
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --split public --attempts 1 --out results/finetune/base-public.json
```

`--adapter PATH` sends the adapter path in the MLX request's `adapters` field. The MLX server maps
`default_model` to its startup configuration, so for a dynamically selected adapter also pass the actual model
path as `--model`:

```bash
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --adapter adapters/qwen35-pilot --split heldout --attempts 1 --out results/finetune/adapter-first.json
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --adapter adapters/qwen35-pilot --split heldout --attempts 3 --out results/finetune/adapter-feedback.json
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --adapter adapters/qwen35-pilot --split heldout --review-notes --out results/finetune/adapter-review.json
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --adapter adapters/qwen35-pilot --split public --attempts 1 --out results/finetune/adapter-public.json
.venv/bin/python tools/ft_eval.py --paired results/finetune/base-first.json results/finetune/adapter-first.json
.venv/bin/python tools/ft_eval.py --paired results/finetune/base-feedback.json results/finetune/adapter-feedback.json
.venv/bin/python tools/ft_eval.py --paired results/finetune/base-review.json results/finetune/adapter-review.json
.venv/bin/python tools/ft_eval.py --paired results/finetune/base-public.json results/finetune/adapter-public.json
```

Do not run training while the inference server is still resident. Stop the server before training, then restart
it for adapter evaluation. The evaluator itself is a client and does not load a model.

## Verification in this workspace

The accepted code and fixture harness passed for all eight held-out tasks:

```text
qmat: PASS
ops: PASS
gdn: PASS
qmat-row: PASS
qgroup: PASS
expert-cache: PASS
predictive: PASS
pinned: PASS
offline checks: 8/8 accepted-code suites passed
```

`make -C engine test` also passed its qgroup, qmat, ops, and GDN fixture suites; `tools/test_autofix.py` passed
3/3 tests. The real-weight checks in [`C_ENGINE.md`](../C_ENGINE.md) were not run because they require MLX/Metal.
The public fixture harness was also checked with reference solutions: all three fixed test suites passed.

The live baseline could not run in this session: the request to `127.0.0.1:8080` returned
`[Errno 1] Operation not permitted`; importing MLX also reports that no Metal device is available. No baseline
model scores are claimed. The first baseline command above is the required next measurement on the interactive
Mac.
