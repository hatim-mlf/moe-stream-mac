/* config.c: strict reader for a qwen3_5_moe config.json. Refuses anything it does not fully understand. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "json.h"
#include "model.h"

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = n > 0 ? malloc((size_t)n + 1) : NULL;
    if (buf && fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); buf = NULL; }
    if (buf) buf[n] = '\0';
    fclose(f);
    return buf;
}

static int num(jval *o, const char *key, double *out) {
    jval *v = json_get(o, key);
    if (!v || v->t != J_NUM) { fprintf(stderr, "config: missing or non-numeric \"%s\"\n", key); return -1; }
    *out = v->num;
    return 0;
}

#define NUM_INT(obj, key, dst) do { double v_; if (num(obj, key, &v_)) goto bad; (dst) = (int)v_; } while (0)
#define NUM_F(obj, key, dst)   do { double v_; if (num(obj, key, &v_)) goto bad; (dst) = (float)v_; } while (0)

static int check_text(jval *t, ModelConfig *c) {
    jval *mt = json_get(t, "model_type"), *rp = json_get(t, "rope_parameters");
    if (!mt || mt->t != J_STR || strncmp(mt->str, "qwen3_5_moe", 11) != 0) {
        fprintf(stderr, "config: model_type must be qwen3_5_moe*, got %s\n", mt && mt->t == J_STR ? mt->str : "?");
        return -1;
    }
    jval *nt = json_get(t, "norm_topk_prob");
    if (nt && nt->t == J_BOOL && !nt->boolean) { fprintf(stderr, "config: norm_topk_prob=false unsupported\n"); return -1; }
    jval *rt = rp ? json_get(rp, "rope_type") : NULL;
    if (!rp || (rt && (rt->t != J_STR || strcmp(rt->str, "default") != 0))) {
        fprintf(stderr, "config: only rope_type \"default\" is supported\n"); return -1;
    }
    float partial;
    NUM_F(rp, "partial_rotary_factor", partial); NUM_F(rp, "rope_theta", c->rope_theta);
    c->rot_dims = (int)(c->head_dim * partial);
    return (c->rot_dims % 2 == 0 && c->rot_dims > 0 && c->rot_dims <= c->head_dim) ? 0 : -1;
bad:
    return -1;
}

int model_config_load(ModelConfig *c, const char *dir) {
    char path[1024]; snprintf(path, sizeof path, "%s/config.json", dir);
    char *text = read_file(path);
    jval *root = text ? json_parse(text, NULL) : NULL;
    int rc = -1;
    if (!root) { fprintf(stderr, "config: cannot read/parse %s\n", path); goto out; }
    jval *t = json_get(root, "text_config"); if (!t) t = root;
    jval *q = json_get(root, "quantization");
    memset(c, 0, sizeof *c);
    NUM_INT(t, "hidden_size", c->hidden);        NUM_INT(t, "num_hidden_layers", c->n_layers);
    NUM_INT(t, "vocab_size", c->vocab);          NUM_INT(t, "full_attention_interval", c->full_attn_interval);
    NUM_INT(t, "num_attention_heads", c->Hq);    NUM_INT(t, "num_key_value_heads", c->Hkv);
    NUM_INT(t, "head_dim", c->head_dim);
    NUM_INT(t, "linear_num_key_heads", c->lin_Hk);   NUM_INT(t, "linear_num_value_heads", c->lin_Hv);
    NUM_INT(t, "linear_key_head_dim", c->lin_Dk);    NUM_INT(t, "linear_value_head_dim", c->lin_Dv);
    NUM_INT(t, "linear_conv_kernel_dim", c->conv_K);
    NUM_INT(t, "num_experts", c->E);             NUM_INT(t, "num_experts_per_tok", c->top_k);
    NUM_INT(t, "moe_intermediate_size", c->moe_inter);
    NUM_INT(t, "shared_expert_intermediate_size", c->sh_inter);
    NUM_F(t, "rms_norm_eps", c->eps);
    if (!q) { fprintf(stderr, "config: no \"quantization\" block (not an MLX quantized checkpoint)\n"); goto out; }
    NUM_INT(q, "group_size", c->group_size);
    jval *mode = json_get(q, "mode");
    if (mode && (mode->t != J_STR || strcmp(mode->str, "affine") != 0)) { fprintf(stderr, "config: only affine quantization\n"); goto out; }
    if (check_text(t, c)) goto out;
    if (c->top_k > MOE_MAX_TOPK || c->lin_Hv % c->lin_Hk || c->Hq % c->Hkv || c->full_attn_interval < 1) {
        fprintf(stderr, "config: unsupported head/expert layout\n"); goto out;
    }
    rc = 0;
bad:
out:
    json_free_tree(root); free(text);
    return rc;
}
