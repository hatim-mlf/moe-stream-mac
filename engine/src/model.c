/* model.c: load a qwen3_5_moe MLX checkpoint and run it one token at a time. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "loader.h"
#include "model.h"
#include "ops.h"

#define P "language_model.model.layers"

static int qm(const Model *m, QMat *dst, int *experts, const char *fmt, int layer, const char *part) {
    char name[320];
    snprintf(name, sizeof name, fmt, layer, part);
    return load_qmat(&m->st, name, m->cfg.group_size, dst, experts);
}

static const float *vec(const Model *m, const char *fmt, int layer, const char *part) {
    char name[320];
    snprintf(name, sizeof name, fmt, layer, part);
    return load_f32(&m->st, name, NULL);
}

static int load_gdn(Model *m, int l, Layer *L) {
    const ModelConfig *c = &m->cfg; GdnWeights *w = &L->gdn;
    const char *f = P ".%d.linear_attn.%s";
    if (qm(m, &w->qkv, NULL, f, l, "in_proj_qkv") || qm(m, &w->z, NULL, f, l, "in_proj_z") ||
        qm(m, &w->a, NULL, f, l, "in_proj_a") || qm(m, &w->b, NULL, f, l, "in_proj_b") ||
        qm(m, &w->out, NULL, f, l, "out_proj")) return -1;
    w->conv_w = vec(m, f, l, "conv1d.weight"); w->A_log = vec(m, f, l, "A_log");
    w->dt_bias = vec(m, f, l, "dt_bias");      w->norm_w = vec(m, f, l, "norm.weight");
    w->hidden = c->hidden; w->Hk = c->lin_Hk; w->Hv = c->lin_Hv; w->Dk = c->lin_Dk; w->Dv = c->lin_Dv;
    w->K = c->conv_K; w->eps = c->eps;
    const int conv_dim = 2 * w->Hk * w->Dk + w->Hv * w->Dv;
    L->gs.conv_state = calloc((size_t)(w->K - 1) * conv_dim, sizeof(float));
    L->gs.rec_state = calloc((size_t)w->Hv * w->Dv * w->Dk, sizeof(float));
    return (w->conv_w && w->A_log && w->dt_bias && w->norm_w && L->gs.conv_state && L->gs.rec_state) ? 0 : -1;
}

static int load_attn(Model *m, int l, Layer *L) {
    const ModelConfig *c = &m->cfg; AttnWeights *w = &L->attn;
    const char *f = P ".%d.self_attn.%s";
    if (qm(m, &w->q, NULL, f, l, "q_proj") || qm(m, &w->k, NULL, f, l, "k_proj") ||
        qm(m, &w->v, NULL, f, l, "v_proj") || qm(m, &w->o, NULL, f, l, "o_proj")) return -1;
    w->q_norm = vec(m, f, l, "q_norm.weight"); w->k_norm = vec(m, f, l, "k_norm.weight");
    w->hidden = c->hidden; w->Hq = c->Hq; w->Hkv = c->Hkv; w->D = c->head_dim; w->rot_dims = c->rot_dims;
    w->rope_theta = c->rope_theta; w->eps = c->eps;
    const size_t kv = (size_t)m->max_len * c->Hkv * c->head_dim;
    L->ac = (AttnCache){ calloc(kv, sizeof(float)), calloc(kv, sizeof(float)), 0, m->max_len };
    return (w->q_norm && w->k_norm && L->ac.k_cache && L->ac.v_cache) ? 0 : -1;
}

static int load_moe(Model *m, int l, Layer *L) {
    const ModelConfig *c = &m->cfg; MoeWeights *w = &L->moe;
    const char *f = P ".%d.mlp.%s";
    char name[320];
    const char *ex[3] = { "switch_mlp.gate_proj", "switch_mlp.up_proj", "switch_mlp.down_proj" };
    ExpertTensor *et[3] = { &w->gate, &w->up, &w->down };
    for (int i = 0; i < 3; i++) {
        snprintf(name, sizeof name, f, l, ex[i]);
        if (expert_tensor_open(et[i], &m->st, name, c->group_size)) return -1;
    }
    if (qm(m, &w->router, NULL, f, l, "gate") ||
        qm(m, &w->sh_gate, NULL, f, l, "shared_expert.gate_proj") || qm(m, &w->sh_up, NULL, f, l, "shared_expert.up_proj") ||
        qm(m, &w->sh_down, NULL, f, l, "shared_expert.down_proj") || qm(m, &w->sh_router, NULL, f, l, "shared_expert_gate"))
        return -1;
    w->hidden = c->hidden; w->E = c->E; w->top_k = c->top_k; w->inter = c->moe_inter; w->sh_inter = c->sh_inter;
    w->layer = l;
    if (w->gate.E != c->E || w->gate.rows != c->moe_inter) { fprintf(stderr, "layer %d: expert shape mismatch\n", l); return -1; }
    /* every layer has the same expert shapes: share one set of read buffers */
    if (l == 0) {
        for (int i = 0; i < c->top_k; i++)
            if (expert_buf_alloc(&w->gbuf[i], &w->gate) || expert_buf_alloc(&w->ubuf[i], &w->up) ||
                expert_buf_alloc(&w->dbuf[i], &w->down)) return -1;
    } else {
        const MoeWeights *w0 = &m->layers[0].moe;
        if (w->gate.w_bytes != w0->gate.w_bytes || w->down.w_bytes != w0->down.w_bytes) {
            fprintf(stderr, "layer %d: expert size differs from layer 0\n", l); return -1;
        }
        for (int i = 0; i < c->top_k; i++) { w->gbuf[i] = w0->gbuf[i]; w->ubuf[i] = w0->ubuf[i]; w->dbuf[i] = w0->dbuf[i]; }
    }
    return 0;
}

static size_t max_scratch(const Model *m) {
    size_t best = 0;
    for (int l = 0; l < m->cfg.n_layers; l++) {
        const Layer *L = &m->layers[l];
        size_t s = L->is_linear ? (size_t)GDN_SCRATCH_FLOATS(&L->gdn) : (size_t)ATTN_SCRATCH_FLOATS(&L->attn, m->max_len);
        size_t e = (size_t)MOE_SCRATCH_FLOATS(&L->moe);
        if (s > best) best = s;
        if (e > best) best = e;
    }
    return best;
}

int model_load(Model *m, const char *dir, int max_len) {
    memset(m, 0, sizeof *m);
    m->max_len = max_len;
    if (model_config_load(&m->cfg, dir) || k3_st_open(&m->st, dir)) return -1;
    const ModelConfig *c = &m->cfg;
    if (load_qmat(&m->st, "language_model.model.embed_tokens", c->group_size, &m->embed, NULL) ||
        load_qmat(&m->st, "language_model.lm_head", c->group_size, &m->lm_head, NULL)) return -1;
    m->final_norm = load_f32(&m->st, "language_model.model.norm.weight", NULL);
    m->layers = calloc((size_t)c->n_layers, sizeof(Layer));
    if (!m->final_norm || !m->layers) return -1;
    for (int l = 0; l < c->n_layers; l++) {
        Layer *L = &m->layers[l];
        L->is_linear = (l + 1) % c->full_attn_interval != 0;
        L->in_norm = vec(m, P ".%d.%s", l, "input_layernorm.weight");
        L->post_norm = vec(m, P ".%d.%s", l, "post_attention_layernorm.weight");
        if (!L->in_norm || !L->post_norm || (L->is_linear ? load_gdn(m, l, L) : load_attn(m, l, L)) || load_moe(m, l, L)) {
            fprintf(stderr, "model: failed to load layer %d\n", l); return -1;
        }
    }
    const size_t H = (size_t)c->hidden;
    m->x = malloc(H * 4); m->h = malloc(H * 4); m->t1 = malloc(H * 4); m->t2 = malloc(H * 4);
    m->scratch = malloc(max_scratch(m) * sizeof(float));
    return (m->x && m->h && m->t1 && m->t2 && m->scratch) ? 0 : -1;
}

void model_reset(Model *m) {
    for (int l = 0; l < m->cfg.n_layers; l++) {
        Layer *L = &m->layers[l];
        if (L->is_linear) {
            const GdnWeights *w = &L->gdn;
            memset(L->gs.conv_state, 0, sizeof(float) * (size_t)(w->K - 1) * (2 * w->Hk * w->Dk + w->Hv * w->Dv));
            memset(L->gs.rec_state, 0, sizeof(float) * (size_t)w->Hv * w->Dv * w->Dk);
        } else {
            L->ac.len = 0;
        }
    }
}

/* One decoder layer: x -> x (in m->x), using m->h, m->t1, m->t2 and m->scratch. */
static int layer_step(Model *m, Layer *L) {
    const int hidden = m->cfg.hidden;
    op_rmsnorm(m->t1, m->x, L->in_norm, hidden, m->cfg.eps);
    if (L->is_linear) {
        if (gdn_layer_step(m->t2, m->t1, &L->gdn, &L->gs, m->scratch)) return -1;
    } else {
        if (attn_layer_step(m->t2, m->t1, &L->attn, &L->ac, m->scratch)) return -1;
    }
    for (int i = 0; i < hidden; i++) m->h[i] = m->x[i] + m->t2[i];
    op_rmsnorm(m->t1, m->h, L->post_norm, hidden, m->cfg.eps);
    if (moe_layer_step(m->t2, m->t1, &L->moe, m->scratch)) return -1;
    for (int i = 0; i < hidden; i++) m->x[i] = m->h[i] + m->t2[i];
    return 0;
}

int model_forward(Model *m, int token, float *logits) {
    if (qmat_row(&m->embed, token, m->x) != 0) return -1;
    for (int l = 0; l < m->cfg.n_layers; l++) {
        if (layer_step(m, &m->layers[l])) return -1;
    }
    op_rmsnorm(m->t1, m->x, m->final_norm, m->cfg.hidden, m->cfg.eps);
    return qmat_matvec(&m->lm_head, m->t1, logits);
}
