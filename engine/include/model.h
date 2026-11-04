/* model.h: the whole Qwen3.5 / Qwen3.6 MoE text model (qwen3_5_moe), MLX affine-quantized checkpoint. */
#ifndef MODEL_H
#define MODEL_H
#include "attn_layer.h"
#include "gdn_layer.h"
#include "k3_st.h"
#include "moe_layer.h"

typedef struct {
    int hidden, n_layers, vocab, full_attn_interval;
    int Hq, Hkv, head_dim, rot_dims;  float rope_theta;
    int lin_Hk, lin_Hv, lin_Dk, lin_Dv, conv_K;
    int E, top_k, moe_inter, sh_inter;
    float eps;
    int group_size;
} ModelConfig;

typedef struct {
    int is_linear;                       /* (layer + 1) % full_attn_interval != 0 */
    const float *in_norm, *post_norm;    /* [hidden] RMSNorm weights */
    GdnWeights gdn;  GdnState gs;        /* used when is_linear  */
    AttnWeights attn; AttnCache ac;      /* used otherwise       */
    MoeWeights moe;
} Layer;

typedef struct Model {
    ModelConfig cfg;
    K3St st;
    QMat embed, lm_head;
    const float *final_norm;             /* [hidden] */
    Layer *layers;
    float *x, *h, *t1, *t2;              /* [hidden] activations */
    float *scratch;                      /* max of the per-layer scratch sizes */
    int max_len;
} Model;

/* Parse <dir>/config.json strictly: every field must be present and supported. 0 or -1 (stderr says why). */
int model_config_load(ModelConfig *c, const char *dir);
/* Open the checkpoint and load every weight into memory. 0 or -1. */
int model_load(Model *m, const char *dir, int max_len);
/* Clear all recurrent / KV state (start a new sequence). */
void model_reset(Model *m);
/* Run one token at the next position; logits gets cfg.vocab floats. 0 or -1. */
int model_forward(Model *m, int token, float *logits);

#endif
