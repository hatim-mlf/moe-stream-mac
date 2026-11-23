# Brief 52: KAT-Coder through pi, then maybe a home-made pi adapter for KAT

Status: planned (2026-11-23). Runs on the Mac (Phase 3). **Starts only after brief 51's full Eval B (Qwen base
vs Qwen + pi adapter) has finished**: KAT and Qwen don't fit in memory together.

## Why

KAT-Coder-V2.5-Dev (OptiQ 4-bit, `~/models/KAT-Coder-V2.5-Dev-OptiQ-4bit`) beats Qwen3.8-27B on our coding
evals and is faster (MoE, ~3B active). The harness-hub pi adapter cannot be used on it: KAT is `qwen3_5_moe`
(40 layers, hidden 2048, 256 experts), Qwen3.8-27B is dense (64 layers, hidden 5120); no tensor shape matches
and `tools/peft_to_mlx.py` refuses it, as it should.

## Phase A: KAT as a third Eval B arm (no training)

1. Add arm `kat` to `tools/pi_agent_eval.py`: model `~/models/KAT-Coder-V2.5-Dev-OptiQ-4bit`, no adapter,
   same sandbox, limits (30 min, 40 tool calls), temperature 0.8 / top-p 1.0, thinking off. Extend
   `summary` to pair every arm against `base` and `kat` vs `pi`. Tests first (`tools/test_pi_agent_eval.py`).
2. Probe before the run, as brief 51 did for Qwen: one pi session on the toy repo through the logging
   proxy. Check that `tool_calls` come back structured (KAT's chat template / mlx-lm tool parser), thinking
   stays off, and the prompt cache is reused between turns. Stop and report if any of these fails.
3. Pilot 2 cases, project the duration, then run all 30. Record per case: pass, wall time, tool calls,
   tokens in/out; per arm: decode and prompt tok/s, peak memory, swap.
4. Report `kat` vs `base`, `kat` vs `pi` with paired CIs, next to KAT's one-shot score on the same 30 cases.

## Phase B: RFT pi adapter for KAT (only if Phase A says it's worth it)

Go only if (a) the pi adapter clearly helped Qwen in brief 51's Eval B (its CI excludes 0, or a large
reduction in turns/time at equal pass rate), and (b) KAT through pi leaves room (fails or hits limits on a
good share of cases). Otherwise stop after Phase A and say so.

The card's recipe (AsyncGRPO on 2×B200) is not possible here. The local version is rejection fine-tuning:

1. Training tasks: code_contests **train** split, none of the 15 eval problems (check by contest id and
   description hash), same stub + public examples + hidden-test verifier as Eval B.
2. Rollouts: KAT through pi, k attempts per task, keep the sessions whose final solution passes the hidden
   tests. Budget before starting: tasks × k × median wall time from Phase A (expect 10+ hours).
3. Export kept sessions as multi-turn tool-call chat data; loss on assistant turns only (check the chat
   template renders tool calls exactly as served; the pi adapter's template uses `{% generation %}` for this).
4. LoRA on KAT with mlx-lm: first check what it can target on `qwen3_5_moe` (attention and shared layers;
   routed experts are SwitchLinear) and that training fits in 32 GB with `tools/mem_guard.py` running.
   Fixture-test any new piece before real weights (CLAUDE.md rule). Reuse `tools/ft_rft.py` where possible.
5. Re-run Eval B arm `kat-rft` on the same 30 cases and pair it against `kat`.

## Record

`results/<UTC>-kat-pi/metrics.json`; delegations in `docs/delegation-log.md`. Numbers next to the llama.cpp
baseline where one exists; say plainly when slower.
