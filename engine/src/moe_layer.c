#include "moe_layer.h"
#include "ops.h"
#include <math.h>
#include <string.h>

typedef struct { float *logits, *probs, *h1, *h2, *h3, *ye, *gsh; } Bufs;

static Bufs carve(const MoeWeights *w, float *scratch) {
    const int m = w->inter > w->sh_inter ? w->inter : w->sh_inter;
    Bufs b;
    b.logits = scratch;   b.probs = b.logits + w->E;  b.h1 = b.probs + w->E;
    b.h2 = b.h1 + m;      b.h3 = b.h2 + m;            b.ye = b.h3 + m;      b.gsh = b.ye + w->hidden;
    return b;
}

/* SwiGLU MLP: y = down( silu(gate x) * up x ), using b->h1, b->h2, b->h3 as temporaries. */
static int mlp(const QMat *gate, const QMat *up, const QMat *down, const float *x, float *y, const Bufs *b) {
    if (qmat_matvec(gate, x, b->h1)) return -1;
    if (qmat_matvec(up, x, b->h2)) return -1;
    op_silu_mul(b->h3, b->h1, b->h2, gate->rows);
    if (qmat_matvec(down, b->h3, y)) return -1;
    return 0;
}

/* Steps 1-2: fill idx[0..top_k) and score[0..top_k). */
static int route(const MoeWeights *w, const float *x, const Bufs *b, int *idx, float *score) {
    if (qmat_matvec(&w->router, x, b->logits)) return -1;
    op_softmax(b->probs, b->logits, w->E);
    op_topk(idx, b->probs, w->E, w->top_k);
    float sum = 0.0f;
    for (int i = 0; i < w->top_k; i++) sum += b->probs[idx[i]];
    for (int i = 0; i < w->top_k; i++) score[i] = b->probs[idx[i]] / sum;
    return 0;
}

int moe_layer_step(float *out, const float *x, const MoeWeights *w, float *scratch) {
    if (w->top_k < 1 || w->top_k > MOE_MAX_TOPK) return -1;
    const Bufs b = carve(w, scratch);
    int idx[MOE_MAX_TOPK];
    float score[MOE_MAX_TOPK];
    if (route(w, x, &b, idx, score)) return -1;
    memset(out, 0, sizeof(float) * (size_t)w->hidden);
    for (int i = 0; i < w->top_k; i++) {
        QMat g, u, d;
        if (expert_tensor_read(&w->gate, idx[i], &w->gbuf, &g) || expert_tensor_read(&w->up, idx[i], &w->ubuf, &u) ||
            expert_tensor_read(&w->down, idx[i], &w->dbuf, &d)) return -1;
        if (mlp(&g, &u, &d, x, b.ye, &b)) return -1;
        for (int j = 0; j < w->hidden; j++) out[j] += score[i] * b.ye[j];
    }
    /* Step 4: shared expert */
    if (mlp(&w->sh_gate, &w->sh_up, &w->sh_down, x, b.ye, &b)) return -1;
    if (qmat_matvec(&w->sh_router, x, b.gsh)) return -1;
    float sg = 1.0f / (1.0f + expf(-b.gsh[0]));
    for (int j = 0; j < w->hidden; j++) out[j] += sg * b.ye[j];
    return 0;
}
