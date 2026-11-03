# Delegation log: Claude (lead) → local Qwen3.6-35B-A3B-4bit (worker)

Protocol: the lead writes the interface, the brief and the fixture-gated test. `tools/delegate.py` sends the brief and stores the code block Qwen returns. The lead compiles and tests it, then sends back the compiler/test output plus a code review until the tests pass. The lead also checks its own spec and fixtures before blaming the worker.

| # | Task | Attempts | What went wrong | Worker time | Result |
|---|---|---|---|---|---|
| 1 | `qmat_matvec`: MLX affine 4/8-bit matvec (`engine/src/qmat.c`) | 3 | 1: used `malloc` (forbidden). 2: invented signed values; never advanced scale/bias per group. Lead's own fixture bug (bf16-rounded reference) found and fixed at the same time | 115 s + 45 s + 57 s | PASS, max err 5e-6 |

Takeaway so far: Qwen follows structure well but drifts from the spec (allocation, invented semantics). It fixes things reliably when the review names the exact bug, and it doesn't find them from raw test output alone.
