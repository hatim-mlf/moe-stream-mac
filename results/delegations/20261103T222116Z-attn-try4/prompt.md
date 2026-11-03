# Task: implement engine/src/attn_layer.c

Write `engine/src/attn_layer.c`, implementing `attn_layer_step` from this header. The numbered steps and the scratch layout in the comments are the spec: follow them exactly.

```c
/* attn_layer.h: one full-attention layer of Qwen3.5 (gated GQA + partial RoPE), one token at a time.
 * Input is the hidden state AFTER input_layernorm; output is the attention branch BEFORE the residual add.
 * Matches mlx_lm.models.qwen3_next.Qwen3NextAttention with a causal mask.
 */
#ifndef ATTN_LAYER_H
#define ATTN_LAYER_H
#include "qmat.h"

typedef struct {
    QMat q;               /* hidden -> Hq * 2D. Per head h the 2D outputs are [query(D) | gate(D)]  */
    QMat k;               /* hidden -> Hkv * D */
    QMat v;               /* hidden -> Hkv * D */
    QMat o;               /* Hq * D -> hidden  */
    const float *q_norm;  /* [D] RMSNorm weight applied to every query head */
    const float *k_norm;  /* [D] RMSNorm weight applied to every key head   */
    int hidden, Hq, Hkv, D, rot_dims;
    float rope_theta, eps;
} AttnWeights;

typedef struct {
    float *k_cache;       /* [max_len][Hkv * D] keys AFTER k_norm and RoPE */
    float *v_cache;       /* [max_len][Hkv * D] */
    int len;              /* tokens cached so far = position of the next token */
    int max_len;
} AttnCache;

/* Scratch layout, in this order: qg [Hq*2D] | q [Hq*D] | att [Hq*D] | kx [Hkv*D] | k [Hkv*D] | vx [Hkv*D] | s [max_len] */
#define ATTN_SCRATCH_FLOATS(w, max_len) (4 * (w)->Hq * (w)->D + 3 * (w)->Hkv * (w)->D + (max_len))

/* out[hidden] = layer(x[hidden]) for the token at position pos = cache->len; then cache->len++.
 * scratch holds ATTN_SCRATCH_FLOATS(w, cache->max_len) floats. Steps:
 *   1. qg = qmat(w->q, x)   [Hq * 2D];  kx = qmat(w->k, x) [Hkv*D];  vx = qmat(w->v, x) [Hkv*D]
 *   2. per head h: query_h = qg + h*2D (D floats), gate_h = qg + h*2D + D (D floats). Gates stay in qg.
 *   3. op_rmsnorm(q + h*D, query_h, w->q_norm, D, eps) per query head;
 *      op_rmsnorm(k + h*D, kx + h*D, w->k_norm, D, eps) per key head.
 *   4. op_rope(q, Hq, D, rot_dims, pos, rope_theta) and op_rope(k, Hkv, D, rot_dims, pos, rope_theta)
 *   5. store k and vx at row pos of k_cache / v_cache
 *   6. for each query head h, kv head kvh = h / (Hq / Hkv):
 *        s[t] = dot(q_h, k_cache[t][kvh]) / sqrt(D)     for t in [0, pos]
 *        p = softmax(s) in place in s: subtract max, exp, sum in double, divide  (n = pos + 1)
 *        att[h*D + i] = sum_t p[t] * v_cache[t][kvh*D + i]
 *   7. att[h*D + i] *= 1 / (1 + exp(-gate_h[i]))   (gate_h read from qg, see step 2)
 *   8. out = qmat(w->o, att)
 * Returns 0; -1 if the cache is full or a qmat_matvec call fails. */
int attn_layer_step(float *out, const float *x, const AttnWeights *w, AttnCache *cache, float *scratch);

#endif
```

Existing, tested functions you must call (do NOT reimplement them):

```c
void op_rmsnorm(float *out, const float *x, const float *w, int n, float eps);
void op_rope(float *x, int n_heads, int head_dim, int rot_dims, int pos, float theta);
int qmat_matvec(const QMat *m, const float *x, float *y);
```

## Requirements

- Portable C99. Include `<math.h>`, `<string.h>`, `"attn_layer.h"`, `"ops.h"`. No allocation, no globals.
- Let QD = Hq*D, KD = Hkv*D. Carve scratch exactly: qg = scratch; q = qg + 2*QD; att = q + QD; kx = att + QD;
  k = kx + KD; vx = k + KD; s = vx + KD.
- Check `cache->len < cache->max_len` FIRST; return -1 if full, before touching anything. pos = cache->len.
- Step 5: memcpy(k_cache + pos*KD, k, KD floats) and memcpy(v_cache + pos*KD, vx, KD floats).
- Step 6 softmax is inline (op_softmax forbids aliasing).
- Check every qmat_matvec return value. Increment cache->len only after success.
- Functions under 50 lines: use static helpers (e.g. project+norm+rope, attend one head, gate+output).

Return only the complete `attn_layer.c` in one fenced code block.


## Your previous attempt failed

The lead engineer has fixed the structure. Fill in the three TODO bodies in this skeleton and return the WHOLE file. Do not change anything outside the TODO bodies.

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
    /* TODO */
}

/* Step 6 for query head h: scores into b->s, softmax in place, weighted sum into b->att + h*D. */
static void attend_head(const AttnWeights *w, const Bufs *b, const AttnCache *c, int h, int pos) {
    /* TODO */
}

/* Steps 7-8: gate b->att with sigmoid(gate_h) read from b->qg, then out = qmat(w->o, b->att). */
static int finish(const AttnWeights *w, const Bufs *b, float *out) {
    /* TODO */
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

Your attempt-2 code had the math right; reuse it. For reference, its body was:

```c
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

Remember: step 3 normalises straight from qg (op_rmsnorm(b->q + h*D, b->qg + h*2*D, ...)), no copy. Each body stays under 50 lines.


Fix it. Reply with the complete corrected file.