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

    /* Shift state: drop the oldest row, append x */
    if (K > 2) {
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