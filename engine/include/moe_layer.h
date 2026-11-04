/* moe_layer.h: Qwen3.5 sparse MoE block, one token. Input is the hidden state AFTER
 * post_attention_layernorm; output is the MLP branch BEFORE the residual add.
 * Matches mlx_lm.models.qwen3_next.Qwen3NextSparseMoeBlock (norm_topk_prob = true).
 */
#ifndef MOE_LAYER_H
#define MOE_LAYER_H
#include "expert_store.h"
#include "qmat.h"

#define MOE_MAX_TOPK 16

typedef struct {
    QMat router;          /* hidden -> E           (mlp.gate)                                 */
    ExpertTensor gate, up, down;   /* routed experts, read from disk on demand (expert_store.h) */
    ExpertBuf gbuf[MOE_MAX_TOPK], ubuf[MOE_MAX_TOPK], dbuf[MOE_MAX_TOPK];  /* one set per selected expert, so the
                                   top_k experts can be read in parallel; may be shared across layers */
    QMat sh_gate, sh_up, sh_down;  /* shared expert: hidden -> sh_inter -> hidden          */
    QMat sh_router;       /* hidden -> 1           (mlp.shared_expert_gate)                   */
    int hidden, E, top_k, inter, sh_inter;
    int layer;            /* for the optional routing trace (env QE_TRACE=<file>) */
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
