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
