# Diagnosis from the delegation records

## Counting rule

The denominator is the 16 repository coding tasks in the delegation log, with 36 Qwen attempts. I count an
attempt as a strict failure when the response failed its tests or needed a lead edit before it was accepted.
By that rule, 27/36 coding attempts failed; only `moe-layer` and `read-sorted` passed on the first attempt
(2/16 = 12.5%). Eight tasks have a logged response accepted without lead code edits, after 17 total model
attempts across those tasks (2.13 attempts per task). `model-forward` is an additional no-edit response that
compiled, but its whole-model check is explicitly pending and is not counted as a confirmed clean pass. This is
historical performance before the current helper rules were sent with every request; it is not a measurement of
the current setup.

The separate coding-prompts request is one additional failed attempt, so the taxonomy below covers 28 failed
folders across 39 model attempts when that non-repository task is included. Task 18 (adding the helper rules)
and the later qmat replay are successful attempts and are not counted as failures. The qmat replay is useful
counter-evidence: with the rules and automatic test feedback, it passed in one attempt, but one replay cannot
establish a general effect.

Each folder is assigned one *primary* blocking class, even where the response had several defects. Links point
to the original attempt folder; each contains the raw prompt, reply, stats, and (where available) system prompt.

## Primary error taxonomy

| Class | Count | Raw failed attempts | Rules/prompt | Automatic test feedback | Training case |
|---|---:|---|---|---|---|
| Quantized signedness or group indexing | 2 | [qmat try2](../../results/delegations/20261103T205215Z-qmat-try2/), [qmat-row try2](../../results/delegations/20261103T232431Z-qmat-row-try2/) | Explicit unsigned-value rule helps; an indexing table or worked example may help more. | Fixture tests caught the errors after compilation. | Repeated mistake makes this a plausible data target, but the qmat replay passed with rules + feedback, so it is not shown to require training. |
| Spec drift, invented behavior, or omitted requested content | 4 | [qmat try1](../../results/delegations/20261103T205114Z-qmat-try1/), [expert-reader try1](../../results/delegations/20261104T133159Z-expert-reader-try1/), [pinned try1](../../results/delegations/20261104T181100Z-pinned-try1/), [coding-prompts](../../results/delegations/20261104T184737Z-coding-prompts/) | Stronger requirement checklist and exact signatures can prevent much of this. | Tests catch behavior only when they encode the omitted requirement; a language/schema check would catch the prompt-list omission. | Training may improve task-faithful completion, but these records do not isolate a training-only failure. |
| Buffer bounds, offsets, or aliasing | 3 | [ops try2](../../results/delegations/20261103T211917Z-ops-try2/), [ops try3](../../results/delegations/20261103T212034Z-ops-try3/), [gdn-layer try1](../../results/delegations/20261103T214838Z-gdn-layer-try1/) | The existing bounds/aliasing rules address these directly; skeletons can lock buffer layout. | Guarded fixtures caught two top-k overrun/tie cases; layer-level fixtures caught projection and alias errors. | Training can reinforce the pattern, but memory-safety review remains necessary. |
| Unused code or compiler hygiene (`-Werror`) | 6 | [ops try1](../../results/delegations/20261103T211754Z-ops-try1/), [gdn try1](../../results/delegations/20261103T214331Z-gdn-try1/), [attn try4](../../results/delegations/20261103T222116Z-attn-try4/), [qmat-row try1](../../results/delegations/20261103T232352Z-qmat-row-try1/), [qgroup try2](../../results/delegations/20261104T123025Z-qgroup-try2/), [fetch try1](../../results/delegations/20261104T124724Z-fetch-try1/) | The current rules explicitly require removing unused code and parenthesizing mixed logical operators. | Compiler feedback catches these immediately and cheaply. | No training-only rationale: this is the clearest case for compile-and-retry automation. |
| API, return-value, or data-shape misuse | 4 | [attn try2](../../results/delegations/20261103T221741Z-attn-try2/), [model-forward try1](../../results/delegations/20261103T232739Z-model-fwd-try1/), [expert-cache try1](../../results/delegations/20261104T142353Z-expert-cache-try1/), [predictive try1](../../results/delegations/20261104T153625Z-predictive-try1/) | Give exact API contracts and return shapes; provide skeletons for multi-stage calls. | Compile/runtime fixtures caught the void-return and data-shape errors. | Repeated contract errors may respond to examples, but a local API map and tests are more direct. |
| Ignored or partly applied review note | 4 | [gdn try2](../../results/delegations/20261103T214445Z-gdn-try2/), [qchat try3](../../results/delegations/20261104T124133Z-qchat-try3/), [expert-reader try2](../../results/delegations/20261104T133250Z-expert-reader-try2/), [predictive try2](../../results/delegations/20261104T153710Z-predictive-try2/) | Rule 9 now says to apply every note and change nothing else. These failures predate that rule being attached to every request. | Test feedback helps when the note maps to a test; it cannot prove every prose note was applied. | This is the strongest candidate for review-following examples, but current-rule compliance must be measured first. |
| Incorrect restructuring/helper decomposition | 1 | [attn try3](../../results/delegations/20261103T221922Z-attn-try3/) | Keep signatures fixed and ask the model to fill a lead-written skeleton; rule 10 supports this. | Compilation and numerical tests found the bad helper/refactor. | A training signal is possible, but the lead-controlled structure is more reliable. |
| Control-token contamination and truncated output | 1 | [qchat try1](../../results/delegations/20261104T123755Z-qchat-try1/) | Do not echo context containing model control tokens; pass only required source context. | A test can reject incomplete output, but cannot recover a truncated response. | Not a useful training target until the prompt/context boundary is fixed. |
| Required behavior omitted | 1 | [qchat try2](../../results/delegations/20261104T123957Z-qchat-try2/) | Enumerate each required option and error path; current helper rules say to follow the brief exactly. | Command tests and sampler-specific cases catch omissions when present. | Fine-tuning could help completeness, but this record alone does not separate it from prompt and test coverage. |
| Incorrect vector/NEON algorithm | 1 | [qgroup try1](../../results/delegations/20261104T122839Z-qgroup-try1/) | A block-level skeleton and explicit lane mapping reduce algorithmic ambiguity. | The 1,800-case fixture suite caught the incorrect vector and scalar paths. | A specialized example could help; testing and a lead-provided loop shape are stronger controls. |
| Lead-side specification defect | 1 | [attn try1](../../results/delegations/20261103T221603Z-attn-try1/) | Fix the brief/header before delegation; this was the lead's inconsistent spec. | No model-side feedback loop can repair a contradictory contract reliably. | Not a training issue. |

The counts sum to 28. The raw evidence agrees with the lead's notes in [`delegation-log.md`](../delegation-log.md): for
example, ops try2 passed its then-current tests but failed review for a top-k overrun/tie defect, and the lead
strengthened the fixtures before try3. This is why the denominator uses strict acceptance rather than “tests
passed once.”

## What the evidence supports

Current helper rules and automatic feedback target most observed classes directly. The only within-project
post-rules signal is the successful qmat replay
([raw replay](../../results/delegations/20261104T194820.409513Z-replay-qmat-a1/)); it supports measuring the
current baseline before training, not assuming the 12.5% historical rate still applies. If a current baseline
still shows repeated semantic or review-note errors after tests and the rules are active, preference and review
examples are a reasonable pilot. No class is proven to be fixable only by training.
