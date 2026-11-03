# Task: implement engine/src/gdn_layer.c

Write `engine/src/gdn_layer.c`, implementing `gdn_layer_step` from this header. The numbered steps in the comment are the spec: implement them in order, exactly.

```c
/* gdn_layer.h: one full Gated DeltaNet layer (Qwen3.5 linear attention), one token at a time.
 * Input is the hidden state AFTER input_layernorm; output is the attention branch BEFORE the
 * residual add. Matches mlx_lm.models.qwen3_5.GatedDeltaNet.__call__.
 */
#ifndef GDN_LAYER_H
#define GDN_LAYER_H
#include "qmat.h"

typedef struct {
    QMat qkv;              /* hidden -> 2*Hk*Dk + Hv*Dv   (q | k | v, in that order)    */
    QMat z;                /* hidden -> Hv*Dv                                            */
    QMat a;                /* hidden -> Hv                                               */
    QMat b;                /* hidden -> Hv                                               */
    QMat out;              /* Hv*Dv  -> hidden                                           */
    const float *conv_w;   /* [conv_dim * K], conv_dim = 2*Hk*Dk + Hv*Dv (gdn_conv_step layout) */
    const float *A_log;    /* [Hv] */
    const float *dt_bias;  /* [Hv] */
    const float *norm_w;   /* [Dv] gated RMSNorm weight, shared by all value heads      */
    int hidden, Hk, Hv, Dk, Dv, K;
    float eps;             /* rms_norm_eps (1e-6) */
} GdnWeights;

typedef struct {
    float *conv_state;     /* [(K-1) * conv_dim], zeros at the start of a sequence */
    float *rec_state;      /* [Hv * Dv * Dk],     zeros at the start of a sequence */
} GdnState;

/* Floats of scratch gdn_layer_step needs. */
#define GDN_SCRATCH_FLOATS(w) (2 * (2 * (w)->Hk * (w)->Dk + (w)->Hv * (w)->Dv) + 2 * (w)->Hv * (w)->Dv + 4 * (w)->Hv)

/* out[hidden] = layer(x[hidden]); advances st by one token. scratch holds GDN_SCRATCH_FLOATS(w) floats.
 * Steps, with conv_dim = 2*Hk*Dk + Hv*Dv:
 *   1. qkv = qmat(w->qkv, x);  z = qmat(w->z, x);  a = qmat(w->a, x);  b = qmat(w->b, x)
 *   2. c = gdn_conv_step(qkv) using st->conv_state and w->conv_w   (c has conv_dim floats)
 *   3. q = c[0 .. Hk*Dk),  k = c[Hk*Dk .. 2*Hk*Dk),  v = c[2*Hk*Dk .. conv_dim)
 *   4. for each key head: gdn_l2norm(q_head, Dk, 1e-6, 1/sqrt(Dk));  gdn_l2norm(k_head, Dk, 1e-6, 1)
 *   5. gdn_gates(g, beta, a, b, w->A_log, w->dt_bias, Hv)
 *   6. y = gdn_recurrent_step(q, k, v, g, beta) using st->rec_state     (y has Hv*Dv floats)
 *   7. for each value head h: y_h = op_rmsnorm(y_h, w->norm_w, Dv, w->eps) * silu(z_h)  elementwise
 *      (silu(z) * normed = op_silu_mul(out, z_h, normed_h, Dv))
 *   8. out = qmat(w->out, y)
 * Returns 0, or -1 if any qmat_matvec call fails. */
int gdn_layer_step(float *out, const float *x, const GdnWeights *w, GdnState *st, float *scratch);

#endif
```

You must call these existing functions (already implemented and tested; do NOT reimplement them):

```c
float bf16_to_f32(uint16_t h);
int qmat_matvec(const QMat *m, const float *x, float *y);
void op_rmsnorm(float *out, const float *x, const float *w, int n, float eps);
void op_silu_mul(float *out, const float *gate, const float *up, int n);
void op_sigmoid(float *out, const float *x, int n);
void op_softmax(float *out, const float *x, int n);
void op_topk(int *idx, const float *x, int n, int k);
void op_rope(float *x, int n_heads, int head_dim, int rot_dims, int pos, float theta);
void gdn_conv_step(float *out, float *state, const float *x, const float *w, int C, int K);
void gdn_l2norm(float *v, int n, float eps, float scale);
void gdn_gates(float *g, float *beta, const float *a, const float *b,
void gdn_recurrent_step(float *y, float *state, const float *q, const float *k, const float *v,
```

## Requirements

- Portable C99. Include `<math.h>`, `"gdn_layer.h"`, `"gdn.h"`, `"ops.h"`. No allocation, no globals.
- Carve every temporary out of `scratch`, in this order: qkv [conv_dim], c [conv_dim], z [Hv*Dv], y [Hv*Dv], a [Hv], b [Hv], g [Hv], beta [Hv].
  That is exactly GDN_SCRATCH_FLOATS. Use `c` for the conv output and normalise q and k inside `c` in place.
- Step 7 needs a temporary for the normed head: write op_rmsnorm's result into the qkv buffer (no longer needed
  by then), then op_silu_mul(y_h, z_h, normed, Dv) writes the gated result back into y_h.
- Check the return value of every qmat_matvec call; return -1 on failure.
- Under 50 lines per function; a static helper for the projections is fine.

Return only the complete `gdn_layer.c` in one fenced code block.


## Your previous attempt failed

```c
#include "gdn_layer.h"
#include "gdn.h"
#include "ops.h"
#include <math.h>

static int project(const float *x, const GdnWeights *w, float *out, int n) {
    return qmat_matvec(&w->qkv, x, out) ||
           qmat_matvec(&w->z, x, out + 2 * w->Hk * w->Dk) ||
           qmat_matvec(&w->a, x, out + 2 * w->Hk * w->Dk + w->Hv * w->Dk) ||
           qmat_matvec(&w->b, x, out + 2 * w->Hk * w->Dk + 2 * w->Hv * w->Dk);
}

int gdn_layer_step(float *out, const float *x, const GdnWeights *w, GdnState *st, float *scratch) {
    int conv_dim = 2 * w->Hk * w->Dk + w->Hv * w->Dv;
    float *c = scratch;
    float *z = c + conv_dim;
    float *y = z + w->Hv * w->Dv;
    float *a = y + w->Hv * w->Dv;
    float *b = a + w->Hv;
    float *g = b + w->Hv;
    float *beta = g + w->Hv;

    if (project(x, w, c, conv_dim) != 0) return -1;

    gdn_conv_step(c, st->conv_state, c, w->conv_w, conv_dim, w->K);

    for (int h = 0; h < w->Hk; h++) {
        gdn_l2norm(c + h * w->Dk, w->Dk, 1e-6, 1.0f / sqrtf((float)w->Dk));
        gdn_l2norm(c + (2 * w->Hk * w->Dk) + h * w->Dk, w->Dk, 1e-6, 1.0f);
    }

    gdn_gates(g, beta, a, b, w->A_log, w->dt_bias, w->Hv);

    gdn_recurrent_step(y, st->rec_state, c, c + 2 * w->Hk * w->Dk, c + 2 * w->Hk * w->Dk + w->Hv * w->Dv, g, beta, w->Hv, w->Dv, w->Dk);

    for (int h = 0; h < w->Hv; h++) {
        float *normed = c;
        float *z_h = z + h * w->Dv;
        float *y_h = y + h * w->Dv;
        op_rmsnorm(normed, y_h, w->norm_w, w->Dv, w->eps);
        op_silu_mul(y_h, z_h, normed, w->Dv);
    }

    return qmat_matvec(&w->out, y, out);
}
```

Compiler output:
```
cc -O2 -std=gnu99 -Wall -Iinclude -Isrc/io -Ithird_party -Werror src/qmat.c src/ops.c src/gdn.c src/loader.c src/io/k3_st.c src/gdn_layer.c tests/test_gdn_layer.c -o build/test_gdn_layer -lm
src/gdn_layer.c:34:135: error: too few arguments to function call, expected 11, have 10
1 error generated.
```

Code review from the lead engineer (several real bugs; rewrite the function following the brief exactly):
1. Scratch layout: use EXACTLY the order in the brief. Let KD = Hk*Dk, VD = Hv*Dv, conv_dim = 2*KD + VD:
     qkv = scratch; c = qkv + conv_dim; z = c + conv_dim; y = z + VD; a = y + VD; b = a + Hv; g = b + Hv; beta = g + Hv;
2. Projections: each one writes to ITS OWN buffer: qmat_matvec(&w->qkv, x, qkv); (&w->z, x, z); (&w->a, x, a); (&w->b, x, b).
   Your `project` wrote z/a/b inside the qkv buffer at wrong offsets.
3. gdn_conv_step must not alias: gdn_conv_step(c, st->conv_state, qkv, w->conv_w, conv_dim, w->K).
   (With out == x it copies its own output into the state.)
4. Inside c: q = c, k = c + KD, v = c + 2*KD. You normalised k at c + 2*KD, which is v.
5. gdn_recurrent_step signature is (y, state, q, k, v, g, beta, Hk, Hv, Dk, Dv) - 11 arguments:
     gdn_recurrent_step(y, st->rec_state, c, c + KD, c + 2*KD, g, beta, w->Hk, w->Hv, w->Dk, w->Dv);
6. Step 7: the normed temporary is `qkv` (free by then), as you did with c - but c still holds q/k/v you no longer need,
   so either is fine once the layout above is used. Keep op_rmsnorm then op_silu_mul(y_h, z_h, normed, Dv).


Fix it. Reply with the complete corrected file.