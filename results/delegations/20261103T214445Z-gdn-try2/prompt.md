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


## Your previous attempt failed

```c
#include "gdn.h"
#include <math.h>
#include <string.h>

/* Helper: stable softplus */
static float softplus(float t) {
    return t > 20.0f ? t : logf(1.0f + expf(t));
}

/* Helper: SiLU */
static float silu(float v) {
    return v / (1.0f + expf(-v));
}

void gdn_conv_step(float *out, float *state, const float *x, const float *w, int C, int K) {
    /* Compute outputs first */
    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        /* Sum over previous inputs: state[j*C + c] * w[c*K + j] for j in [0, K-1) */
        for (int j = 0; j < K - 1; j++) {
            sum += state[j * C + c] * w[c * K + j];
        }
        /* Add current input contribution: w[c*K + K-1] * x[c] */
        sum += w[c * K + K - 1] * x[c];
        out[c] = silu(sum);
    }

    /* Shift state: move last K-2 rows up by one row, then copy x into last row */
    /* State is (K-1) rows of C elements. We need to shift rows 0 to K-3 up to rows 0 to K-3,
     * and copy x into row K-2 (the last row). */
    if (K > 2) {
        /* Move rows 0..K-3 to rows 0..K-3 (they stay in place for the first K-2 rows)
         * Actually, we need to shift: old row 1 becomes new row 0, old row 2 becomes new row 1, ...,
         * old row K-2 becomes new row K-3, and x becomes new row K-2.
         * So we memmove rows 1..K-2 to rows 0..K-3. */
        int shift_rows = K - 2; /* number of rows to shift up */
        int row_size = C;
        /* Shift: copy row j+1 to row j for j in 0..K-3 */
        for (int j = 0; j < shift_rows; j++) {
            memcpy(state + j * C, state + (j + 1) * C, C * sizeof(float));
        }
    }
    /* Copy x into the last row (row K-2) */
    memcpy(state + (K - 2) * C, x, C * sizeof(float));
}

void gdn_l2norm(float *v, int n, float eps, float scale) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        sum_sq += (double)v[i] * (double)v[i];
    }
    float norm = (float)(scale / sqrt(sum_sq + (double)eps));
    for (int i = 0; i < n; i++) {
        v[i] *= norm;
    }
}

void gdn_gates(float *g, float *beta, const float *a, const float *b,
               const float *A_log, const float *dt_bias, int Hv) {
    for (int h = 0; h < Hv; h++) {
        /* g[h] = exp( -exp(A_log[h]) * softplus(a[h] + dt_bias[h]) ) */
        float sp = softplus(a[h] + dt_bias[h]);
        g[h] = expf(-expf(A_log[h]) * sp);

        /* beta[h] = 1 / (1 + exp(-b[h])) */
        beta[h] = 1.0f / (1.0f + expf(-b[h]));
    }
}

void gdn_recurrent_step(float *y, float *state, const float *q, const float *k, const float *v,
                        const float *g, const float *beta, int Hk, int Hv, int Dk, int Dv) {
    /* state is [Hv][Dv][Dk] row-major: state[h*Dv*Dk + i*Dk + j] */
    /* For each value head h, compute key head kh = h / (Hv / Hk) */
    int Hk_per_Hv = Hv / Hk; /* number of value heads per key head */

    for (int h = 0; h < Hv; h++) {
        int kh = h / Hk_per_Hv; /* key head index */
        const float *kq = q + kh * Dk; /* key query for this key head */
        const float *kk = k + kh * Dk; /* key key for this key head */

        /* State pointer for value head h: state[h*Dv*Dk] */
        float *S = state + h * Dv * Dk;

        /* For each row i in [0, Dv) */
        for (int i = 0; i < Dv; i++) {
            float *S_row = S + i * Dk;

            /* Step 1: Decay S[i][j] *= g[h] for all j in [0, Dk) */
            for (int j = 0; j < Dk; j++) {
                S_row[j] *= g[h];
            }

            /* Step 2: kv[i] = sum_j S[i][j] * kk[j] */
            float kv_i = 0.0f;
            for (int j = 0; j < Dk; j++) {
                kv_i += S_row[j] * kk[j];
            }

            /* Step 3: d[i] = (v[h*Dv + i] - kv[i]) * beta[h] */
            float d_i = (v[h * Dv + i] - kv_i) * beta[h];

            /* Step 4: S[i][j] += kk[j] * d_i for all j in [0, Dk) */
            for (int j = 0; j < Dk; j++) {
                S_row[j] += kk[j] * d_i;
            }

            /* Step 5: y[h*Dv + i] = sum_j S[i][j] * kq[j] */
            float y_i = 0.0f;
            for (int j = 0; j < Dk; j++) {
                y_i += S_row[j] * kq[j];
            }
            y[h * Dv + i] = y_i;
        }
    }
}
```

Compiler output (-Wall -Wextra -Werror):
```
cc -O2 -std=c99 -Wall -Wextra -Werror -Iinclude src/gdn.c tests/test_gdn.c -o build/test_gdn -lm
src/gdn.c:37:13: error: unused variable 'row_size' [-Werror,-Wunused-variable]
1 error generated.
```

Review: all numerical tests pass when that warning is relaxed, so the logic is correct. Two cleanups only:
1. Delete the unused `row_size` variable.
2. In gdn_conv_step, replace the long "Actually, we need to shift..." comment block with one line: /* drop the oldest row, append x */
Do not change any computation.


Fix it. Reply with the complete corrected file.