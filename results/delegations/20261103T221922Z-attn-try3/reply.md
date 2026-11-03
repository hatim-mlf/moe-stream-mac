```c
#include "attn_layer.h"
#include "ops.h"
#include <math.h>
#include <string.h>

/* Helper: steps 1-5. Projects x into scratch, normalizes, applies RoPE, caches keys/values.
 * Returns 0 on success, -1 if any qmat_matvec fails. */
static int prepare(const QMat *q, const QMat *k, const QMat *v, const float *x,
                   const float *q_norm, const float *k_norm,
                   float *qg, float *q, float *kx, float *k, float *vx,
                   float *k_cache, float *v_cache, int *len, int max_len,
                   int Hq, int Hkv, int D, int pos, int rot_dims, float rope_theta, float eps) {
    if (len[0] >= max_len) return -1;
    pos = len[0];
    if (qmat_matvec(q, x, qg) != 0) return -1;
    if (qmat_matvec(k, x, kx) != 0) return -1;
    if (qmat_matvec(v, x, vx) != 0) return -1;

    int h;
    for (h = 0; h < Hq; h++) {
        /* Step 3: RMSNorm query heads directly from qg into q */
        op_rmsnorm(q + h * D, qg + h * 2 * D, q_norm, D, eps);
    }
    for (h = 0; h < Hkv; h++) {
        /* Step 3: RMSNorm key heads directly from kx into k */
        op_rmsnorm(k + h * D, kx + h * D, k_norm, D, eps);
    }

    /* Step 4: RoPE */
    op_rope(q, Hq, D, rot_dims, pos, rope_theta);
    op_rope(k, Hkv, D, rot_dims, pos, rope_theta);

    /* Step 5: Cache keys and values */
    memcpy(k_cache + pos * (Hkv * D), k, (size_t)(Hkv * D) * sizeof(float));
    memcpy(v_cache + pos * (Hkv * D), vx, (size_t)(Hkv * D) * sizeof(float));

    len[0]++;
    return 0;
}

/* Helper: Step 6 for one query head h. Computes attention weights and output for this head.
 * Writes softmax probabilities into s, and the weighted sum of values into att_h. */
static void attend_head(const float *q_h, int h, int Hq, int Hkv, int D, int pos,
                        const float *k_cache, const float *v_cache, float *s, float *att_h) {
    int kvh = h / (Hq / Hkv);
    int t, i;
    int n = pos + 1;

    /* Compute scores */
    for (t = 0; t <= pos; t++) {
        const float *k_t = k_cache + t * (Hkv * D) + kvh * D;
        float score = 0.0f;
        for (i = 0; i < D; i++) score += q_h[i] * k_t[i];
        s[t] = score / sqrtf((float)D);
    }

    /* Softmax: subtract max, exp, sum, divide */
    float max_val = -INFINITY;
    for (t = 0; t <= pos; t++) {
        if (s[t] > max_val) max_val = s[t];
    }
    double sum_exp = 0.0;
    for (t = 0; t <= pos; t++) {
        sum_exp += exp(s[t] - max_val);
    }
    for (t = 0; t <= pos; t++) {
        s[t] = (float)(exp(s[t] - max_val) / sum_exp);
    }

    /* Compute weighted sum of values */
    for (i = 0; i < D; i++) {
        float val = 0.0f;
        for (t = 0; t <= pos; t++) {
            const float *v_t = v_cache + t * (Hkv * D) + kvh * D;
            val += s[t] * v_t[i];
        }
        att_h[i] = val;
    }
}

/* Helper: Steps 7-8. Applies gate to attention output and projects to final output. */
static int finish(const QMat *o, const float *qg, const float *att, float *out,
                  int Hq, int D) {
    int h, i;
    /* Step 7: Apply gate */
    for (h = 0; h < Hq; h++) {
        const float *gate_h = qg + h * 2 * D + D;
        float *att_h = att + h * D;
        for (i = 0; i < D; i++) {
            float gate = gate_h[i];
            float inv_gate = 1.0f / (1.0f + expf(-gate));
            att_h[i] *= inv_gate;
        }
    }

    /* Step 8: Output projection */
    return qmat_matvec(o, att, out);
}

int attn_layer_step(float *out, const float *x, const AttnWeights *w, AttnCache *cache, float *scratch) {
    int Hq = w->Hq, Hkv = w->Hkv, D = w->D;
    int QD = Hq * D, KD = Hkv * D;
    int pos;
    int len = cache->len;

    /* Scratch layout: qg [Hq*2D] | q [Hq*D] | att [Hq*D] | kx [Hkv*D] | k [Hkv*D] | vx [Hkv*D] | s [max_len] */
    float *qg = scratch;
    float *q = qg + 2 * QD;
    float *att = q + QD;
    float *kx = att + QD;
    float *k = kx + KD;
    float *vx = k + KD;
    float *s = vx + KD;

    /* Step 1-5: Prepare, normalize, rope, cache */
    if (prepare(&w->q, &w->k, &w->v, x, w->q_norm, w->k_norm,
                qg, q, kx, k, vx,
                cache->k_cache, cache->v_cache, &len, cache->max_len,
                Hq, Hkv, D, pos, w->rot_dims, w->rope_theta, w->eps) != 0) {
        return -1;
    }

    /* Step 6: Attention for each query head */
    int h;
    for (h = 0; h < Hq; h++) {
        float *q_h = q + h * D;
        float *att_h = att + h * D;
        attend_head(q_h, h, Hq, Hkv, D, len - 1, cache->k_cache, cache->v_cache, s, att_h);
    }

    /* Step 7-8: Gate and output projection */
    if (finish(&w->o, qg, att, out, Hq, D) != 0) {
        return -1;
    }

    cache->len = len;
    return 0;
}
```