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
