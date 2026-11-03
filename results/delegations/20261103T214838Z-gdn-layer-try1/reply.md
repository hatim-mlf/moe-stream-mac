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