# Task: implement engine/src/gdn.c

Write `engine/src/gdn.c`, implementing every function declared in this header (`#include "gdn.h"`). The comments are the spec: follow every formula and memory layout exactly.

```c
/* gdn.h: Gated DeltaNet (Qwen3.5 linear attention) pieces, one token at a time. All float32.
 * Matches mlx_lm.models.qwen3_5.GatedDeltaNet / gated_delta.py. Buffers never alias unless stated.
 */
#ifndef GDN_H
#define GDN_H

/* Causal depthwise conv over the current input and the previous K-1 inputs, then SiLU.
 *   x:     [C]            current input (one channel per c)
 *   w:     [C * K]        w[c*K + j]; j = K-1 multiplies the CURRENT input, j = 0 the oldest
 *   state: [(K-1) * C]    previous inputs, oldest first: state[j*C + c] is input t-(K-1)+j
 *   out[c] = silu( sum_{j<K-1} w[c*K + j] * state[j*C + c]  +  w[c*K + K-1] * x[c] )
 *   silu(v) = v / (1 + exp(-v))
 * Afterwards state holds the last K-1 inputs including x (drop the oldest, append x).
 * A fresh state is all zeros. K >= 2. */
void gdn_conv_step(float *out, float *state, const float *x, const float *w, int C, int K);

/* In place: v[i] = scale * v[i] / sqrt(sum_j v[j]^2 + eps)   (sum in double) */
void gdn_l2norm(float *v, int n, float eps, float scale);

/* Per value head h in [0, Hv):
 *   g[h]    = exp( -exp(A_log[h]) * softplus(a[h] + dt_bias[h]) )
 *   beta[h] = 1 / (1 + exp(-b[h]))
 *   softplus(t) = log(1 + exp(t)), computed stably: t > 20 ? t : log1p(exp(t)) */
void gdn_gates(float *g, float *beta, const float *a, const float *b,
               const float *A_log, const float *dt_bias, int Hv);

/* One recurrent step for all Hv value heads (Hv is a multiple of Hk).
 *   q, k:  [Hk * Dk]   already normalised;  v: [Hv * Dv];  g, beta: [Hv]
 *   state: [Hv][Dv][Dk] row-major, updated in place;  y: [Hv * Dv]
 * Value head h uses key head kh = h / (Hv / Hk). With S = state[h], kq = q[kh*Dk..], kk = k[kh*Dk..]:
 *   S[i][j] *= g[h]                                  for all i < Dv, j < Dk
 *   kv[i]    = sum_j S[i][j] * kk[j]
 *   d[i]     = (v[h*Dv + i] - kv[i]) * beta[h]
 *   S[i][j] += kk[j] * d[i]
 *   y[h*Dv + i] = sum_j S[i][j] * kq[j]
 * Per-row i the steps are independent, so one pass per row is fine: decay, dot with kk, update, dot with kq. */
void gdn_recurrent_step(float *y, float *state, const float *q, const float *k, const float *v,
                        const float *g, const float *beta, int Hk, int Hv, int Dk, int Dv);

#endif
```

## Requirements

- Portable C99. Only `<math.h>`, `<string.h>` and `"gdn.h"`. No allocation, no globals, no static buffers.
- `gdn_conv_step`: compute all of `out` first, THEN shift the state (memmove the last K-2 rows up one row,
  copy x into the last row). The state is [(K-1)][C]: row j, column c is `state[j*C + c]`.
- `gdn_recurrent_step`: loop heads, then rows i of S. For each row: multiply by g, take the dot with kk,
  compute d, add kk*d, then take the dot with kq. Accumulate dots in float.
- Functions under 50 lines; small static helpers allowed. Brief comments only where non-obvious.

Return only the complete `gdn.c` in one fenced code block.
