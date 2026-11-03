/* test_gdn_layer: full GatedDeltaNet layer on real weights vs tests/gdn_layer_ref.py (MLX).
 *   test_gdn_layer <model dir> <layer> <ref.bin> */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "gdn_layer.h"
#include "loader.h"

#define GROUP 64
#define REL_TOL 2e-3f   /* MLX runs parts of the layer in bf16 on the GPU */

static int load_layer(const K3St *s, int layer, GdnWeights *w) {
    char base[256], name[320];
    snprintf(base, sizeof base, "language_model.model.layers.%d.linear_attn", layer);
    const char *proj[5] = { "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj" };
    QMat *dst[5] = { &w->qkv, &w->z, &w->a, &w->b, &w->out };
    for (int i = 0; i < 5; i++) {
        snprintf(name, sizeof name, "%s.%s", base, proj[i]);
        if (load_qmat(s, name, GROUP, dst[i], NULL)) return -1;
    }
    snprintf(name, sizeof name, "%s.conv1d.weight", base); w->conv_w = load_f32(s, name, NULL);
    snprintf(name, sizeof name, "%s.A_log", base);         w->A_log = load_f32(s, name, NULL);
    snprintf(name, sizeof name, "%s.dt_bias", base);       w->dt_bias = load_f32(s, name, NULL);
    snprintf(name, sizeof name, "%s.norm.weight", base);   w->norm_w = load_f32(s, name, NULL);
    if (!w->conv_w || !w->A_log || !w->dt_bias || !w->norm_w) return -1;
    /* Qwen3.6-35B-A3B text config */
    w->hidden = 2048; w->Hk = 16; w->Hv = 32; w->Dk = 128; w->Dv = 128; w->K = 4; w->eps = 1e-6f;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s model layer ref.bin\n", argv[0]); return 2; }
    K3St s; GdnWeights w;
    if (k3_st_open(&s, argv[1]) || load_layer(&s, atoi(argv[2]), &w)) return 1;
    FILE *f = fopen(argv[3], "rb"); int32_t hdr[2];
    if (!f || fread(hdr, 4, 2, f) != 2 || hdr[1] != w.hidden) { fprintf(stderr, "bad ref file\n"); return 1; }
    const int T = hdr[0], H = hdr[1], conv_dim = 2 * w.Hk * w.Dk + w.Hv * w.Dv;
    float *x = malloc(sizeof(float) * T * H), *want = malloc(sizeof(float) * T * H), *got = malloc(sizeof(float) * H);
    if (fread(x, 4, (size_t)T * H, f) != (size_t)T * H || fread(want, 4, (size_t)T * H, f) != (size_t)T * H) return 1;
    fclose(f);
    GdnState st = { calloc((size_t)(w.K - 1) * conv_dim, 4), calloc((size_t)w.Hv * w.Dv * w.Dk, 4) };
    float *scratch = malloc(sizeof(float) * GDN_SCRATCH_FLOATS(&w));
    int failed = 0;
    for (int t = 0; t < T; t++) {
        if (gdn_layer_step(got, x + t * H, &w, &st, scratch)) { printf("FAIL  step %d returned error\n", t); return 1; }
        double num = 0, den = 0; float worst = 0;
        for (int i = 0; i < H; i++) {
            float e = fabsf(got[i] - want[t * H + i]); if (e > worst) worst = e;
            num += (double)e * e; den += (double)want[t * H + i] * want[t * H + i];
        }
        float rel = (float)sqrt(num / (den + 1e-30));
        failed |= !(rel < REL_TOL);
        printf("%s  token %d: relative L2 err %.2e, max abs err %.2e\n", rel < REL_TOL ? "PASS" : "FAIL", t, rel, worst);
    }
    printf(failed ? "GDN LAYER TEST FAILED\n" : "GDN LAYER MATCHES MLX\n");
    return failed;
}
