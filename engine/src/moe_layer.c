#include "moe_layer.h"
#include "ops.h"
#include <math.h>
#include <string.h>
#include <time.h>

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

/* Read the gate, up and down matrices of the top_k selected experts idx[0..top_k) into their own buffers
 * (w->gbuf[i], w->ubuf[i], w->dbuf[i]) IN PARALLEL, and return QMat views in g[i], u[i], d[i].
 * Returns 0, or -1 if any read failed. */
static int fetch_experts(const MoeWeights *w, const int *idx, QMat *g, QMat *u, QMat *d) {
    int failed = 0;
    #pragma omp parallel for schedule(dynamic, 1) reduction(|:failed)
    for (int j = 0; j < 3 * w->top_k; ++j) {
        int i = j / 3;
        int which = j % 3;
        int expert = idx[i];
        const ExpertBuf *buf = (which == 0) ? &w->gbuf[i] : (which == 1) ? &w->ubuf[i] : &w->dbuf[i];
        const ExpertTensor *t = (which == 0) ? &w->gate : (which == 1) ? &w->up : &w->down;
        QMat *out = (which == 0) ? &g[i] : (which == 1) ? &u[i] : &d[i];
        failed |= (expert_tensor_read(t, expert, buf, out) != 0);
    }
    return failed ? -1 : 0;
}

int moe_layer_step(float *out, const float *x, const MoeWeights *w, float *scratch) {
    if (w->top_k < 1 || w->top_k > MOE_MAX_TOPK) return -1;
    const Bufs b = carve(w, scratch);
    int idx[MOE_MAX_TOPK];
    float score[MOE_MAX_TOPK];
    if (route(w, x, &b, idx, score)) return -1;
    QMat g[MOE_MAX_TOPK], u[MOE_MAX_TOPK], d[MOE_MAX_TOPK];
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (fetch_experts(w, idx, g, u, d)) return -1;          /* all top_k experts read from disk in parallel */
    clock_gettime(CLOCK_MONOTONIC, &t1);
    expert_add_read_seconds((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9);
    memset(out, 0, sizeof(float) * (size_t)w->hidden);
    for (int i = 0; i < w->top_k; i++) {
        if (mlp(&g[i], &u[i], &d[i], x, b.ye, &b)) return -1;
        for (int j = 0; j < w->hidden; j++) out[j] += score[i] * b.ye[j];
    }
    /* Step 4: shared expert */
    if (mlp(&w->sh_gate, &w->sh_up, &w->sh_down, x, b.ye, &b)) return -1;
    if (qmat_matvec(&w->sh_router, x, b.gsh)) return -1;
    float sg = 1.0f / (1.0f + expf(-b.gsh[0]));
    for (int j = 0; j < w->hidden; j++) out[j] += sg * b.ye[j];
    return 0;
}
