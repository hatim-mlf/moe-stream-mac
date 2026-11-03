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

#define ATTN_SCRATCH_FLOATS(w, max_len) (2 * (w)->Hq * (w)->D + (w)->Hq * (w)->D + 2 * (w)->Hkv * (w)->D + (max_len))

/* out[hidden] = layer(x[hidden]) for the token at position pos = cache->len; then cache->len++.
 * scratch holds ATTN_SCRATCH_FLOATS(w, cache->max_len) floats. Steps:
 *   1. qg = qmat(w->q, x)   [Hq * 2D];  kx = qmat(w->k, x) [Hkv*D];  vx = qmat(w->v, x) [Hkv*D]
 *   2. split qg per head h: query_h = qg[h*2D .. h*2D + D),  gate_h = qg[h*2D + D .. h*2D + 2D)
 *      Copy the queries into a contiguous q [Hq*D] buffer and the gates into gate [Hq*D].
 *   3. op_rmsnorm each query head with w->q_norm and each key head with w->k_norm (eps = w->eps), in place
 *      is NOT allowed by op_rmsnorm (no aliasing), so normalise into the destination buffer.
 *   4. op_rope(q, Hq, D, rot_dims, pos, rope_theta) and op_rope(k, Hkv, D, rot_dims, pos, rope_theta)
 *   5. store k and vx at row pos of k_cache / v_cache
 *   6. for each query head h, kv head kvh = h / (Hq / Hkv):
 *        s[t] = dot(q_h, k_cache[t][kvh]) / sqrt(D)     for t in [0, pos]
 *        p = op_softmax(s)
 *        att_h[i] = sum_t p[t] * v_cache[t][kvh*D + i]
 *   7. att[j] *= sigmoid(gate[j])  for all j < Hq*D
 *   8. out = qmat(w->o, att)
 * Returns 0; -1 if the cache is full or a qmat_matvec call fails. */
int attn_layer_step(float *out, const float *x, const AttnWeights *w, AttnCache *cache, float *scratch);

#endif
```

Existing, tested functions you must call (do NOT reimplement them):

```c
void op_rmsnorm(float *out, const float *x, const float *w, int n, float eps);
void op_silu_mul(float *out, const float *gate, const float *up, int n);
void op_sigmoid(float *out, const float *x, int n);
void op_softmax(float *out, const float *x, int n);
void op_topk(int *idx, const float *x, int n, int k);
void op_rope(float *x, int n_heads, int head_dim, int rot_dims, int pos, float theta);
float bf16_to_f32(uint16_t h);
int qmat_matvec(const QMat *m, const float *x, float *y);
```

## Requirements

- Portable C99. Include `<math.h>`, `<string.h>`, `"attn_layer.h"`, `"ops.h"`. No allocation, no globals.
- Carve the scratch buffers with pointer arithmetic in exactly the documented order. Let QD = Hq*D, KD = Hkv*D.
- Check `cache->len < cache->max_len` FIRST; return -1 if full, before touching anything.
- op_softmax does not allow aliasing: compute scores into s, then call op_softmax(s, s, n)? No - that aliases.
  Instead implement step 6's softmax inline: subtract the max, exponentiate, sum in double, divide. n = pos + 1.
- Store into the cache with memcpy: k_cache + pos*KD gets k, v_cache + pos*KD gets vx. Increment len at the end.
- Check every qmat_matvec return value. Functions under 50 lines: use static helpers (e.g. one per step group).

Return only the complete `attn_layer.c` in one fenced code block.
