# Task: implement engine/src/moe_layer.c by filling in a skeleton

The lead engineer has fixed the structure. Fill in the three TODOs (the bodies of `mlp` and `route`, and the shared-expert step at the end of `moe_layer_step`). Return the WHOLE file. Do not change anything outside the TODOs.

Header (the spec):

```c
/* moe_layer.h: Qwen3.5 sparse MoE block, one token. Input is the hidden state AFTER
 * post_attention_layernorm; output is the MLP branch BEFORE the residual add.
 * Matches mlx_lm.models.qwen3_next.Qwen3NextSparseMoeBlock (norm_topk_prob = true).
 */
#ifndef MOE_LAYER_H
#define MOE_LAYER_H
#include "qmat.h"

#define MOE_MAX_TOPK 16

typedef struct {
    QMat router;          /* hidden -> E           (mlp.gate)                                 */
    QMat gate, up, down;  /* ALL experts: [E][inter][hidden] / [E][hidden][inter]; view one with qmat_expert() */
    QMat sh_gate, sh_up, sh_down;  /* shared expert: hidden -> sh_inter -> hidden          */
    QMat sh_router;       /* hidden -> 1           (mlp.shared_expert_gate)                   */
    int hidden, E, top_k, inter, sh_inter;
} MoeWeights;

/* Scratch layout: logits [E] | probs [E] | h1 [max(inter, sh_inter)] | h2 [same] | h3 [same] | ye [hidden] | gsh [1] */
#define MOE_SCRATCH_FLOATS(w) (2 * (w)->E + 3 * ((w)->inter > (w)->sh_inter ? (w)->inter : (w)->sh_inter) + (w)->hidden + 1)

/* out[hidden] = moe(x). Steps:
 *   1. logits = qmat(router, x);  probs = op_softmax(logits);  idx = op_topk(probs, top_k)
 *   2. score_i = probs[idx_i] / sum_j probs[idx_j]                  (renormalised over the top_k)
 *   3. out = 0;  for each i: ye = expert_mlp(idx_i, x);  out += score_i * ye
 *        expert_mlp(e, x) = down_e( silu(gate_e x) * up_e x )     (op_silu_mul(h3, h1=gate_e x, h2=up_e x))
 *   4. s = sh_down( silu(sh_gate x) * sh_up x );  out += sigmoid(qmat(sh_router, x)[0]) * s
 * Returns 0, or -1 if top_k > MOE_MAX_TOPK or any qmat_matvec fails. */
int moe_layer_step(float *out, const float *x, const MoeWeights *w, float *scratch);

#endif
```

Skeleton to complete:

```c
#include "moe_layer.h"
#include "loader.h"   /* qmat_expert() */
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
    /* TODO */
}

/* Steps 1-2: fill idx[0..top_k) and score[0..top_k). */
static int route(const MoeWeights *w, const float *x, const Bufs *b, int *idx, float *score) {
    /* TODO */
}

int moe_layer_step(float *out, const float *x, const MoeWeights *w, float *scratch) {
    if (w->top_k < 1 || w->top_k > MOE_MAX_TOPK) return -1;
    const Bufs b = carve(w, scratch);
    int idx[MOE_MAX_TOPK];
    float score[MOE_MAX_TOPK];
    if (route(w, x, &b, idx, score)) return -1;
    memset(out, 0, sizeof(float) * (size_t)w->hidden);
    for (int i = 0; i < w->top_k; i++) {
        const QMat g = qmat_expert(&w->gate, idx[i]), u = qmat_expert(&w->up, idx[i]), d = qmat_expert(&w->down, idx[i]);
        if (mlp(&g, &u, &d, x, b.ye, &b)) return -1;
        for (int j = 0; j < w->hidden; j++) out[j] += score[i] * b.ye[j];
    }
    /* Step 4: shared expert */
    /* TODO: s into b.ye via mlp(); gate scalar via qmat(sh_router) into b.gsh; out += sigmoid(gsh[0]) * s */
    return 0;
}
```

Existing, tested functions (call them, do not reimplement):

```c
int qmat_matvec(const QMat *m, const float *x, float *y);
QMat qmat_expert(const QMat *all, int e);
void op_silu_mul(float *out, const float *gate, const float *up, int n);
void op_sigmoid(float *out, const float *x, int n);
void op_softmax(float *out, const float *x, int n);
void op_topk(int *idx, const float *x, int n, int k);
```

## Requirements

- `mlp`: h1 = gate x, h2 = up x, h3 = silu(h1) * h2 via op_silu_mul(b->h3, b->h1, b->h2, gate->rows), then y = down h3.
  Check each qmat_matvec; return -1 on failure, 0 otherwise.
- `route`: logits = qmat(&w->router, x) into b->logits; op_softmax(b->probs, b->logits, w->E);
  op_topk(idx, b->probs, w->E, w->top_k); score[i] = probs[idx[i]] / (sum over the top_k of probs[idx[j]]).
- Shared expert: mlp(&w->sh_gate, &w->sh_up, &w->sh_down, x, b.ye, &b); qmat(&w->sh_router, x) into b.gsh;
  float sg = 1 / (1 + expf(-b.gsh[0])); out[j] += sg * b.ye[j].
- No new variables that go unused (the build uses -Werror).

Return only the complete `moe_layer.c` in one fenced code block.
