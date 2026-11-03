/* test_gdn: gdn.h against fixtures from tools/gen_gdn_fixture.py (MLX ground truth). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gdn.h"

#define MAX_ARRAYS 40
#define TOL 2e-5f

typedef struct { int n; float *a[MAX_ARRAYS]; int len[MAX_ARRAYS]; } Fix;

static int load(const char *dir, const char *name, Fix *f) {
    char path[512]; snprintf(path, sizeof path, "%s/gdn_%s.bin", dir, name);
    FILE *fp = fopen(path, "rb"); memset(f, 0, sizeof *f);
    if (!fp || fread(&f->n, 4, 1, fp) != 1 || f->n > MAX_ARRAYS) { if (fp) fclose(fp); return -1; }
    for (int i = 0; i < f->n; i++) {
        if (fread(&f->len[i], 4, 1, fp) != 1) { fclose(fp); return -1; }
        f->a[i] = malloc(sizeof(float) * f->len[i]);
        if (fread(f->a[i], 4, f->len[i], fp) != (size_t)f->len[i]) { fclose(fp); return -1; }
    }
    fclose(fp); return 0;
}
static void drop(Fix *f) { for (int i = 0; i < f->n; i++) free(f->a[i]); }

static int compare(const char *name, const float *got, const float *want, int n) {
    float worst = 0.f; int bad = 0;
    for (int i = 0; i < n; i++) {
        float e = fabsf(got[i] - want[i]);
        if (e > worst) worst = e;
        if (!(e <= TOL + TOL * fabsf(want[i]))) bad++;
    }
    printf("%s  %-14s max err %.2e (%d/%d off)\n", bad ? "FAIL" : "PASS", name, worst, bad, n);
    return bad != 0;
}

static int test_conv(const char *dir) {
    Fix f; if (load(dir, "conv", &f)) return 1;
    const int C = (int)f.a[0][0], K = (int)f.a[0][1], T = (int)f.a[0][2];
    float *state = calloc((size_t)(K - 1) * C, sizeof(float)), *out = malloc(sizeof(float) * C);
    int failed = 0; char name[32];
    for (int t = 0; t < T; t++) {
        gdn_conv_step(out, state, f.a[2] + t * C, f.a[1], C, K);
        snprintf(name, sizeof name, "conv t%d", t);
        failed |= compare(name, out, f.a[3] + t * C, C);
    }
    free(state); free(out); drop(&f); return failed;
}

static int test_l2_gates(const char *dir) {
    Fix f; int failed = 0;
    if (load(dir, "l2norm", &f)) return 1;
    const int H = (int)f.a[0][0], Dk = (int)f.a[0][1];
    float *v = malloc(sizeof(float) * H * Dk);
    memcpy(v, f.a[1], sizeof(float) * H * Dk);
    for (int h = 0; h < H; h++) gdn_l2norm(v + h * Dk, Dk, 1e-6f, 1.0f / sqrtf((float)Dk));
    failed |= compare("l2norm q", v, f.a[2], H * Dk);
    memcpy(v, f.a[3], sizeof(float) * H * Dk);
    for (int h = 0; h < H; h++) gdn_l2norm(v + h * Dk, Dk, 1e-6f, 1.0f);
    failed |= compare("l2norm k", v, f.a[4], H * Dk);
    free(v); drop(&f);
    if (load(dir, "gates", &f)) return 1;
    const int Hv = (int)f.a[0][0];
    float g[64], beta[64];
    gdn_gates(g, beta, f.a[1], f.a[2], f.a[3], f.a[4], Hv);
    failed |= compare("gates g", g, f.a[5], Hv) | compare("gates beta", beta, f.a[6], Hv);
    drop(&f); return failed;
}

static int test_recurrence(const char *dir) {
    Fix f; if (load(dir, "recurrence", &f)) return 1;
    const float *p = f.a[0];
    const int Hk = (int)p[0], Hv = (int)p[1], Dk = (int)p[2], Dv = (int)p[3], T = (int)p[4];
    float *state = calloc((size_t)Hv * Dv * Dk, sizeof(float)), *y = malloc(sizeof(float) * Hv * Dv);
    int failed = 0; char name[32];
    for (int t = 0; t < T; t++) {
        float **a = f.a + 1 + t * 6;  /* q k v g beta y */
        gdn_recurrent_step(y, state, a[0], a[1], a[2], a[3], a[4], Hk, Hv, Dk, Dv);
        snprintf(name, sizeof name, "recur t%d", t);
        failed |= compare(name, y, a[5], Hv * Dv);
    }
    failed |= compare("recur state", state, f.a[1 + T * 6], Hv * Dv * Dk);
    free(state); free(y); drop(&f); return failed;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/fixtures";
    int failed = test_conv(dir) | test_l2_gates(dir) | test_recurrence(dir);
    printf(failed ? "GDN TESTS FAILED\n" : "ALL GDN TESTS PASSED\n");
    return failed;
}
