#include "gdn_layer.h"
#include "gdn.h"
#include "ops.h"
#include <math.h>

int gdn_layer_step(float *out, const float *x, const GdnWeights *w, GdnState *st, float *scratch) {
    int KD = w->Hk * w->Dk;
    int VD = w->Hv * w->Dv;
    int conv_dim = 2 * KD + VD;

    /* Scratch layout per brief:
     * qkv [conv_dim], c [conv_dim], z [VD], y [VD], a [Hv], b [Hv], g [Hv], beta [Hv]
     */
    float *qkv = scratch;
    float *c = qkv + conv_dim;
    float *z = c + conv_dim;
    float *y = z + VD;
    float *a = y + VD;
    float *b = a + w->Hv;
    float *g = b + w->Hv;
    float *beta = g + w->Hv;

    /* Step 1: projections into their own buffers */
    if (qmat_matvec(&w->qkv, x, qkv) != 0) return -1;
    if (qmat_matvec(&w->z, x, z) != 0) return -1;
    if (qmat_matvec(&w->a, x, a) != 0) return -1;
    if (qmat_matvec(&w->b, x, b) != 0) return -1;

    /* Step 2: convolution */
    gdn_conv_step(c, st->conv_state, qkv, w->conv_w, conv_dim, w->K);

    /* Step 3: split c into q, k, v */
    /* q = c, k = c + KD, v = c + 2*KD */

    /* Step 4: L2 normalize q and k heads */
    for (int h = 0; h < w->Hk; h++) {
        gdn_l2norm(c + h * w->Dk, w->Dk, 1e-6, 1.0f / sqrtf((float)w->Dk));
        gdn_l2norm(c + KD + h * w->Dk, w->Dk, 1e-6, 1.0f);
    }

    /* Step 5: gates */
    gdn_gates(g, beta, a, b, w->A_log, w->dt_bias, w->Hv);

    /* Step 6: recurrent step */
    gdn_recurrent_step(y, st->rec_state, c, c + KD, c + 2 * KD, g, beta, w->Hk, w->Hv, w->Dk, w->Dv);

    /* Step 7: gated RMSNorm + SiLU */
    /* Use qkv buffer as temporary for normed output (no longer needed) */
    for (int h = 0; h < w->Hv; h++) {
        float *normed = qkv;
        float *z_h = z + h * w->Dv;
        float *y_h = y + h * w->Dv;
        op_rmsnorm(normed, y_h, w->norm_w, w->Dv, w->eps);
        op_silu_mul(y_h, z_h, normed, w->Dv);
    }

    /* Step 8: output projection */
    return qmat_matvec(&w->out, y, out);
}
