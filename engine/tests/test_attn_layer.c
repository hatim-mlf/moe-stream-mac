/* test_attn_layer: full-attention layer on real weights vs tests/attn_layer_ref.py (MLX).
 *   test_attn_layer <model dir> <layer> <ref.bin> */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "attn_layer.h"
#include "loader.h"

#define GROUP 64
#define REL_TOL 2e-3f

static int load_layer(const K3St *s, int layer, AttnWeights *w) {
    char base[256], name[320];
    snprintf(base, sizeof base, "language_model.model.layers.%d.self_attn", layer);
    const char *proj[4] = { "q_proj", "k_proj", "v_proj", "o_proj" };
    QMat *dst[4] = { &w->q, &w->k, &w->v, &w->o };
    for (int i = 0; i < 4; i++) {
        snprintf(name, sizeof name, "%s.%s", base, proj[i]);
        if (load_qmat(s, name, GROUP, dst[i], NULL)) return -1;
    }
    snprintf(name, sizeof name, "%s.q_norm.weight", base); w->q_norm = load_f32(s, name, NULL);
    snprintf(name, sizeof name, "%s.k_norm.weight", base); w->k_norm = load_f32(s, name, NULL);
    if (!w->q_norm || !w->k_norm) return -1;
    /* Qwen3.6-35B-A3B text config */
    w->hidden = 2048; w->Hq = 16; w->Hkv = 2; w->D = 256; w->rot_dims = 64;
    w->rope_theta = 1e7f; w->eps = 1e-6f;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s model layer ref.bin\n", argv[0]); return 2; }
    K3St s; AttnWeights w;
    if (k3_st_open(&s, argv[1]) || load_layer(&s, atoi(argv[2]), &w)) return 1;
    FILE *f = fopen(argv[3], "rb"); int32_t hdr[2];
    if (!f || fread(hdr, 4, 2, f) != 2 || hdr[1] != w.hidden) { fprintf(stderr, "bad ref file\n"); return 1; }
    const int T = hdr[0], H = hdr[1], kvd = w.Hkv * w.D;
    float *x = malloc(sizeof(float) * T * H), *want = malloc(sizeof(float) * T * H), *got = malloc(sizeof(float) * H);
    if (fread(x, 4, (size_t)T * H, f) != (size_t)T * H || fread(want, 4, (size_t)T * H, f) != (size_t)T * H) return 1;
    fclose(f);
    AttnCache c = { calloc((size_t)T * kvd, 4), calloc((size_t)T * kvd, 4), 0, T };
    float *scratch = malloc(sizeof(float) * ATTN_SCRATCH_FLOATS(&w, T));
    int failed = 0;
    for (int t = 0; t < T; t++) {
        if (attn_layer_step(got, x + t * H, &w, &c, scratch)) { printf("FAIL  step %d returned error\n", t); return 1; }
        double num = 0, den = 0;
        for (int i = 0; i < H; i++) {
            double e = got[i] - want[t * H + i]; num += e * e; den += (double)want[t * H + i] * want[t * H + i];
        }
        float rel = (float)sqrt(num / (den + 1e-30));
        failed |= !(rel < REL_TOL);
        printf("%s  token %d: relative L2 err %.2e\n", rel < REL_TOL ? "PASS" : "FAIL", t, rel);
    }
    float extra[1];
    int full = attn_layer_step(extra, x, &w, &c, scratch);   /* cache is full now: must refuse */
    printf("%s  full cache refused\n", full == -1 ? "PASS" : "FAIL"); failed |= full != -1;
    printf(failed ? "ATTN LAYER TEST FAILED\n" : "ATTN LAYER MATCHES MLX\n");
    return failed;
}
