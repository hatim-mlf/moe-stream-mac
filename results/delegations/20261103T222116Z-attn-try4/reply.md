```c
#include "attn_layer.h"
#include "ops.h"
#include <math.h>
#include <string.h>

typedef struct { float *qg, *q, *att, *kx, *k, *vx, *s; } Bufs;

static Bufs carve(const AttnWeights *w, float *scratch) {
    const int QD = w->Hq * w->D, KD = w->Hkv * w->D;
    Bufs b;
    b.qg = scratch;      b.q = b.qg + 2 * QD; b.att = b.q + QD;
    b.kx = b.att + QD;   b.k = b.kx + KD;     b.vx = b.k + KD;   b.s = b.vx + KD;
    return b;
}

/* Steps 1-5 for the token at position pos. Returns 0, or -1 if a qmat_matvec fails. */
static int prepare(const AttnWeights *w, const float *x, const Bufs *b, AttnCache *c, int pos) {
    const int QD = w->Hq * w->D, KD = w->Hkv * w->D;
    const int Hq = w->Hq, Hkv = w->Hkv, D = w->D;
    /* Step 1: project */
    if (qmat_matvec(&w->q, x, b->qg) != 0) return -1;
    if (qmat_matvec(&w->k, x, b->kx) != 0) return -1;
    if (qmat_matvec(&w->v, x, b->vx) != 0) return -1;

    /* Step 3: RMSNorm (in-place on q, reading from qg) */
    for (int h = 0; h < Hq; h++) {
        op_rmsnorm(b->q + h * D, b->qg + h * 2 * D, w->q_norm, D, w->eps);
    }
    for (int h = 0; h < Hkv; h++) {
        op_rmsnorm(b->k + h * D, b->kx + h * D, w->k_norm, D, w->eps);
    }

    /* Step 4: RoPE */
    op_rope(b->q, Hq, D, w->rot_dims, pos, w->rope_theta);
    op_rope(b->k, Hkv, D, w->rot_dims, pos, w->rope_theta);

    /* Step 5: store k and vx in cache */
    memcpy(c->k_cache + pos * KD, b->k, KD * sizeof(float));
    memcpy(c->v_cache + pos * KD, b->vx, KD * sizeof(float));

    return 0;
}

/* Step 6 for query head h: scores into b->s, softmax in place, weighted sum into b->att + h*D. */
static void attend_head(const AttnWeights *w, const Bufs *b, const AttnCache *c, int h, int pos) {
    const int D = w->D, KD = w->Hkv * w->D;
    const int kvh = h / (w->Hq / w->Hkv);
    const float *q_h = b->q + h * D;
    float *att_h = b->att + h * D;

    /* Compute scores */
    for (int t = 0; t <= pos; t++) {
        float score = 0.0f;
        const float *k_t = c->k_cache + t * KD + kvh * D;
        for (int i = 0; i < D; i++) score += q_h[i] * k_t[i];
        b->s[t] = score / sqrtf((float)D);
    }

    /* Softmax in place */
    float max_val = -INFINITY;
    for (int t = 0; t <= pos; t++) {
        if (b->s[t] > max_val) max_val = b->s[t];
    }
    double sum_exp = 0.0;
    for (int t = 0; t <= pos; t++) {
        sum_exp += exp(b->s[t] - max_val);
    }
    for (int t = 0; t <= pos; t++) {
        b->s[t] = (float)(exp(b->s[t] - max_val) / sum_exp);
    }

    /* Weighted sum into att_h */
    for (int i = 0; i < D; i++) {
        float val = 0.0f;
        for (int t = 0; t <= pos; t++) {
            const float *v_t = c->v_cache + t * KD + kvh * D;
            val += b->s[t] * v_t[i];
        }
        att_h[i] = val;
    }
}

/* Steps 7-8: gate b->att with sigmoid(gate_h) read from b->qg, then out = qmat(w->o, b->att). */
static int finish(const AttnWeights *w, const Bufs *b, float *out) {
    const int QD = w->Hq * w->D, D = w->D;
    /* Step 7: apply gate */
    for (int h = 0; h < w->Hq; h++) {
        const float *gate_h = b->qg + h * 2 * D + D;
        float *att_h = b->att + h * D;
        for (int i = 0; i < D; i++) {
            float gate = gate_h[i];
            float inv_gate = 1.0f / (1.0f + expf(-gate));
            att_h[i] *= inv_gate;
        }
    }

    /* Step 8: output projection */
    return qmat_matvec(&w->o, b->att, out);
}

int attn_layer_step(float *out, const float *x, const AttnWeights *w, AttnCache *cache, float *scratch) {
    if (cache->len >= cache->max_len) return -1;
    const int pos = cache->len;
    const Bufs b = carve(w, scratch);
    if (prepare(w, x, &b, cache, pos)) return -1;
    for (int h = 0; h < w->Hq; h++) attend_head(w, &b, cache, h, pos);
    if (finish(w, &b, out)) return -1;
    cache->len = pos + 1;
    return 0;
}
```