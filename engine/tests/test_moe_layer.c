/* test_moe_layer: sparse MoE block on real weights vs tests/moe_layer_ref.py (MLX). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "loader.h"
#include "moe_layer.h"

#define GROUP 64
#define REL_TOL 2e-3f

static int load_layer(const K3St *s, int layer, MoeWeights *w) {
    char name[320];
    const char *parts[8] = { "gate", "switch_mlp.gate_proj", "switch_mlp.up_proj", "switch_mlp.down_proj",
                             "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj",
                             "shared_expert_gate" };
    QMat *dst[8] = { &w->router, &w->gate, &w->up, &w->down, &w->sh_gate, &w->sh_up, &w->sh_down, &w->sh_router };
    int experts = 0;
    for (int i = 0; i < 8; i++) {
        snprintf(name, sizeof name, "language_model.model.layers.%d.mlp.%s", layer, parts[i]);
        if (load_qmat(s, name, GROUP, dst[i], i == 1 ? &experts : NULL)) return -1;
    }
    /* Qwen3.6-35B-A3B text config */
    w->hidden = 2048; w->E = experts; w->top_k = 8; w->inter = 512; w->sh_inter = 512;
    return experts == 256 && w->gate.rows == 512 ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s model layer ref.bin\n", argv[0]); return 2; }
    K3St s; MoeWeights w;
    if (k3_st_open(&s, argv[1]) || load_layer(&s, atoi(argv[2]), &w)) { fprintf(stderr, "load failed\n"); return 1; }
    FILE *f = fopen(argv[3], "rb"); int32_t hdr[2];
    if (!f || fread(hdr, 4, 2, f) != 2 || hdr[1] != w.hidden) { fprintf(stderr, "bad ref file\n"); return 1; }
    const int T = hdr[0], H = hdr[1];
    float *x = malloc(sizeof(float) * T * H), *want = malloc(sizeof(float) * T * H), *got = malloc(sizeof(float) * H);
    if (fread(x, 4, (size_t)T * H, f) != (size_t)T * H || fread(want, 4, (size_t)T * H, f) != (size_t)T * H) return 1;
    fclose(f);
    float *scratch = malloc(sizeof(float) * MOE_SCRATCH_FLOATS(&w));
    int failed = 0;
    for (int t = 0; t < T; t++) {
        if (moe_layer_step(got, x + t * H, &w, scratch)) { printf("FAIL  token %d returned error\n", t); return 1; }
        double num = 0, den = 0;
        for (int i = 0; i < H; i++) {
            double e = got[i] - want[t * H + i]; num += e * e; den += (double)want[t * H + i] * want[t * H + i];
        }
        float rel = (float)sqrt(num / (den + 1e-30));
        failed |= !(rel < REL_TOL);
        printf("%s  token %d: relative L2 err %.2e\n", rel < REL_TOL ? "PASS" : "FAIL", t, rel);
    }
    MoeWeights bad = w; bad.top_k = MOE_MAX_TOPK + 1;
    int refused = moe_layer_step(got, x, &bad, scratch) == -1;
    printf("%s  top_k > MOE_MAX_TOPK refused\n", refused ? "PASS" : "FAIL"); failed |= !refused;
    printf(failed ? "MOE LAYER TEST FAILED\n" : "MOE LAYER MATCHES MLX\n");
    return failed;
}
