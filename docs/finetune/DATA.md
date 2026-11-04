# Dataset pipeline

## Splits and natural log examples

Run:

```bash
.venv/bin/python tools/ft_build_dataset.py
```

The builder reads the raw prompt/reply folders, cross-checks the final accepted source with `git show`, and
extracts the brief, first failed code, accepted code, and latest review prompt. Natural task assignments are:

| Split | IDs |
|---|---|
| Train | `gdn-layer`, `attn-layer`, `moe-layer`, `qchat`, `fetch-experts`, `read-sorted` |
| Validation only | `model-forward`, `expert-reader` |
| Held out: never emitted to any dataset | `qmat`, `ops`, `gdn`, `qmat-row`, `qgroup`, `expert-cache`, `predictive`, `pinned` |

For natural SFT, a user message contains the original brief and the assistant message contains the accepted source
from the listed final Git revision. For preference training, `preferences.jsonl` stores the system/user message
list in `prompt`, and one-element assistant-message lists in `chosen` and `rejected` for accepted and first failed
code. It contains only train IDs. These conversational prompt/chosen/rejected records are sidecars; the installed
`mlx_lm.lora` CLI does not train DPO or ORPO.

For review following, `review_follow.jsonl` pairs the most recent retry prompt (including the prior code and
review/test feedback) with the accepted correction. Three train review rows are appended to `train.jsonl`; two
validation review rows are appended to `valid.jsonl`. The combined review sidecar is retained for inspection.
The natural extraction map and source revisions are in `tools/ft_build_dataset.py`, and hashes/splits/counts are
recorded in `data/finetune/manifest.json`.

`model-forward` stays validation-only; the delegation log says it compiled but its whole-model comparison was
still pending. Treat its validation loss and probe example as shape/loader checks, not a correctness label, until
the real-model gate in `C_ENGINE.md` is run.

## New execution-verified tasks

The checked-in seed bank currently contains 28 engineer-authored tasks: 10 C99/skeleton kernels, 12 Python
stdlib/skeleton utilities, and 6 NumPy utilities. Each reference solution is executed against fixed assertions;
C is compiled with `cc -std=c99 -Wall -Wextra -Werror` and fixture assertions, and Python runs in the project
`.venv`. Six seed examples go to seed validation and 22 to seed training. These seeds are useful for checking the
loader/trainer and are not represented as stronger-teacher outputs.

For scale, the builder can ask a stronger OpenAI-compatible teacher to create one task at a time across C99
kernels, Python stdlib/NumPy utilities, and skeleton-fill tasks. It accepts a candidate only after compiling or
executing its supplied tests. Failed candidates and failure output go to `rejected.jsonl`; accepted examples are
assigned to train/valid by task ID. A production seed-growth target is 250 accepted teacher examples, producing
roughly 280 total synthetic examples before natural log examples:

```bash
FT_TEACHER_API_KEY=<key-if-required> .venv/bin/python tools/ft_build_dataset.py \
  --teacher-url https://<trusted-teacher>/v1/chat/completions \
  --teacher-model <stronger-coding-model> --teacher-count 250 --max-rejections 5
```

Use only a trusted teacher endpoint and isolated, disposable working environment: rejection sampling executes
teacher-provided C/Python tests locally and is not an OS security sandbox. No teacher was configured in this
phase and no external model/data was downloaded.

## Files and MLX format

`train.jsonl` and `valid.jsonl` contain one JSON object per row with a chat `messages` array (system, user,
assistant), which is the format read by the installed MLX-LM `ChatDataset`. The extra `task_id` field is ignored
by that reader and enables the leakage assertion. I verified this against
`.venv/lib/python3.14/site-packages/mlx_lm/tuner/datasets.py` in the installed 0.32.2 package. The builder also writes:

- `seed_only/train.jsonl` and `seed_only/valid.jsonl` for seed-only loader checks.
- `probe/train.jsonl` (four seed rows plus the longest train-eligible review example) and `probe/valid.jsonl`
  (the longest validation review example) for a five-step memory probe that includes real task context length.
- `preferences.jsonl` for later DPO/ORPO tooling.
- `review_follow.jsonl` for direct review-example inspection.
- `rejected.jsonl` for teacher candidates that failed their own tests.
- `manifest.json` with task split, source hashes, validation counts, and row counts.

Before writing, `verify_no_heldout` checks train, validation, preference, and review rows against the held-out ID
set. Public HumanEval checks are defined only in the evaluator and never enter this pipeline.

## Builder output verified here

The actual command above produced:

```json
{
  "seed_validated": 28,
  "teacher_accepted": 0,
  "teacher_rejected": 0,
  "seed_only_train_rows": 22,
  "seed_only_valid_rows": 6,
  "probe_train_rows": 5,
  "probe_valid_rows": 1,
  "train_rows": 31,
  "valid_rows": 10,
  "preference_rows": 3,
  "review_train_rows": 3,
  "review_valid_rows": 2
}
```

The final `train.jsonl` and `valid.jsonl` were also inspected by `task_id`: no held-out task appears in either
file. The local tokenizer was loaded from the existing model directory without loading weights or downloading
anything. Applying its chat template gave these maximum sequence lengths: `train.jsonl` 5,269 tokens,
`valid.jsonl` 9,153, `seed_only` train/valid 584/543, and the memory probe train/valid 5,269/9,153. The recipe
therefore uses 10,240 tokens; recheck lengths after adding teacher tasks and before training. Teacher generation
remains an unrun growth step because no stronger teacher endpoint was configured. The rejection loop was
smoke-checked with a local deterministic stub: it rejected one failing Python candidate, then accepted a passing
Python task and C99 task (`accepted=2`, `rejected=1`). This does not substitute for generating the planned teacher set.
