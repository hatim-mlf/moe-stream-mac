```c
#include "attn_layer.h"
#include "ops.h"
#include <math.h>
#include <string.h>

/* Helper: project x, then RMSNorm+RoPE the result into dst. */
static int project_norm_rope(const QMat *m, const float *x, const float *norm_w,
                             float *dst, int H, int D, int pos, float theta) {
    if (qmat_matvec(m, x, dst) != 0) return -1;
    op_rmsnorm(dst, dst, norm_w, D, 0.0f); /* eps unused by op_rmsnorm */
    op_rope(dst, H, D, 0, pos, theta);
    return 0;
}

/* Helper: attend one query head h, using kv head kvh. */
static void attend_one_head(float *att, const float *q_h, const float *k_cache,
                            const float *v_cache, const float *s, int pos, int kvh, int D) {
    int t, i;
    float sum = 0.0f;
    for (t = 0; t <= pos; t++) {
        const float *k_t = k_cache + t * (Hkv * D) + kvh * D;
        for (i = 0; i < D; i++) sum += q_h[i] * k_t[i];
    }
    sum /= sqrtf((float)D);
    /* softmax inline: subtract max, exp, sum, divide */
    float max_val = -INFINITY;
    for (t = 0; t <= pos; t++) {
        float val = 0.0f;
        const float *k_t = k_cache + t * (Hkv * D) + kvh * D;
        for (i = 0; i < D; i++) val += q_h[i] * k_t[i];
        val /= sqrtf((float)D);
        if (val > max_val) max_val = val;
    }
    double sum_exp = 0.0;
    for (t = 0; t <= pos; t++) {
        float val = 0.0f;
        const float *k_t = k_cache + t * (Hkv * D) + kvh * D;
        for (i = 0; i < D; i++) val += q_h[i] * k_t[i];
        val /= sqrtf((float)D);
        sum_exp += exp(val - max_val);
    }
    for (i = 0; i < D; i++) {
        float val = 0.0f;
        for (t = 0; t <= pos; t++) {
            const float *k_t = k_cache + t * (Hkv * D) + kvh * D;
            const float *v_t = v_cache + t * (Hkv * D) + kvh * D;
            val += s[t] * v_t[i];
        }
        att[i] = val / (float)sum_exp;
    }
}

int attn_layer_step(float *out, const float *x, const AttnWeights *w, AttnCache *cache, float *scratch) {
    if (cache->len >= cache->max_len) return -1;
    int pos = cache->len;
    int Hq = w->Hq, Hkv = w->Hkv, D = w->D;
    int QD = Hq * D, KD = Hkv * D;
    int rot_dims = w->rot_dims;
    float rope_theta = w->rope_theta;

    /* Scratch layout: qg [Hq*2D] | q [Hq*D] | att [Hq*D] | kx [Hkv*D] | k [Hkv*D] | vx [Hkv*D] | s [max_len] */
    float *qg = scratch;
    float *q = qg + 2 * QD;
    float *att = q + QD;
    float *kx = att + QD;
    float *k = kx + KD;
    float *vx = k + KD;
    float *s = vx + KD;

    /* Step 1: project */
    if (qmat_matvec(&w->q, x, qg) != 0) return -1;
    if (qmat_matvec(&w->k, x, kx) != 0) return -1;
    if (qmat_matvec(&w->v, x, vx) != 0) return -1;

    /* Step 2: split qg into query and gate */
    int h, kvh;
    for (h = 0; h < Hq; h++) {
        float *query_h = qg + h * 2 * D;
        float *gate_h = query_h + D;
        float *q_h = q + h * D;
        memcpy(q_h, query_h, D * sizeof(float));
        /* gate_h stays in qg, we'll use it later */
    }

    /* Step 3: RMSNorm */
    for (h = 0; h < Hq; h++) {
        float *q_h = q + h * D;
        op_rmsnorm(q_h, q_h, w->q_norm, D, w->eps);
    }
    for (h = 0; h < Hkv; h++) {
        float *k_h = k + h * D;
        op_rmsnorm(k_h, kx + h * D, w->k_norm, D, w->eps);
    }

    /* Step 4: RoPE */
    op_rope(q, Hq, D, rot_dims, pos, rope_theta);
    op_rope(k, Hkv, D, rot_dims, pos, rope_theta);

    /* Step 5: store k and vx in cache */
    memcpy(cache->k_cache + pos * KD, k, KD * sizeof(float));
    memcpy(cache->v_cache + pos * KD, vx, KD * sizeof(float));

    /* Step 6: attention */
    for (h = 0; h < Hq; h++) {
        kvh = h / (Hq / Hkv);
        float *q_h = q + h * D;
        float *att_h = att + h * D;
        /* Compute scores */
        for (int t = 0; t <= pos; t++) {
            float score = 0.0f;
            const float *k_t = cache->k_cache + t * KD + kvh * D;
            for (int i = 0; i < D; i++) score += q_h[i] * k_t[i];
            score /= sqrtf((float)D);
            s[t] = score;
        }
        /* Softmax */
        float max_val = -INFINITY;
        for (int t = 0; t <= pos; t++) {
            if (s[t] > max_val) max_val = s[t];
        }
        double sum_exp = 0.0;
        for (int t = 0; t <= pos; t++) {
            sum_exp += exp(s[t] - max_val);
        }
        for (int t = 0; t <= pos; t++) {
            s[t] = (float)(exp(s[t] - max_val) / sum_exp);
        }
        /* Compute output */
        for (int i = 0; i < D; i++) {
            float val = 0.0f;
            for (int t = 0; t <= pos; t++) {
                const float *v_t = cache->v_cache + t * KD + kvh * D;
                val += s[t] * v_t[i];
            }
            att_h[i] = val;
        }
    }

    /* Step 7: apply gate */
    for (h = 0; h < Hq; h++) {
        float *gate_h = qg + h * 2 * D + D;
        float *att_h = att + h * D;
        for (int i = 0; i < D; i++) {
            float gate = gate_h[i];
            float inv_gate = 1.0f / (1.0f + expf(-gate));
            att_h[i] *= inv_gate;
        }
    }

    /* Step 8: output projection */
    if (qmat_matvec(&w->o, att, out) != 0) return -1;

    cache->len++;
    return 0;
}
```