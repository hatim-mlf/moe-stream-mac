# Delegation log: Claude (lead) → local Qwen3.6-35B-A3B-4bit (worker)

Protocol: the lead writes the interface, the brief and the fixture-gated test. `tools/delegate.py` sends the brief and stores the code block Qwen returns. The lead compiles and tests it, then sends back the compiler/test output plus a code review until the tests pass. The lead also checks its own spec and fixtures before blaming the worker.

| # | Task | Attempts | What went wrong | Worker time | Result |
|---|---|---|---|---|---|
| 1 | `qmat_matvec`: MLX affine 4/8-bit matvec (`engine/src/qmat.c`) | 3 | 1: used `malloc` (forbidden). 2: invented signed values; never advanced scale/bias per group. Lead's own fixture bug (bf16-rounded reference) found and fixed at the same time | 115 s + 45 s + 57 s | PASS, max err 5e-6 |

Takeaway so far: Qwen follows structure well but drifts from the spec (allocation, invented semantics). It fixes things reliably when the review names the exact bug, and it doesn't find them from raw test output alone.
| 2 | safetensors reader for MLX checkpoints | — | Done by lead: reused upstream `k3_st` (two small edits: U32 dtype, rank 8). Checked on 6 real 35B layers vs `mx.quantized_matmul`, max rel err 6e-6 | — | PASS |
| 3 | `ops.c`: rmsnorm, silu·mul, sigmoid, softmax, top-k, RoPE | 3 + 1 lead fix | 1: left a dead wrong `inv_freq` line (-Werror). 2: passed the tests, but review found a top-k buffer overrun and reversed ties, so the lead strengthened the tests. 3: fixed both. Lead then found a third overrun (tie with the k-th value) and fixed it with one character after adding a test | 122 s + 72 s + 53 s | PASS, 11 checks |

Lessons: passing tests ≠ correct. Review every worker diff for memory safety, and turn each review finding into a test before sending it back.
| 4a | `gdn.c`: conv step, l2norm, gates, delta-rule recurrence | 3 | 1: numerically right on the first try, but an unused variable broke -Werror and it left long "thinking aloud" comments. 2: **ignored the review** and kept the variable. 3: complied once the exact line was quoted | 128 s + 57 s + 51 s | PASS, 16 checks |
| 4b | `gdn_layer.c`: full Gated DeltaNet layer (projections → conv → norm → gates → recurrence → gated norm → out) | 2 | 1: lost track of buffers. z/a/b written inside qkv at wrong offsets, conv in/out aliased, k at v's offset, recurrence missing Hk. 2: fixed all five from a line-by-line review | 78 s + 50 s | PASS on real 35B weights, layers 0 and 34, rel err ~2e-6 vs MLX |

Lead wrote: `loader.c` (real-checkpoint glue), every spec, test and MLX reference.
