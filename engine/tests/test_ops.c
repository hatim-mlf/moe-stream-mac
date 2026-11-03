/* test_ops: ops.h against fixtures from tools/gen_ops_fixture.py (MLX ground truth). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ops.h"

#define MAX_ARRAYS 8
#define TOL 1e-5f

typedef struct { int n; float *a[MAX_ARRAYS]; int len[MAX_ARRAYS]; } Fix;

static int load(const char *dir, const char *name, Fix *f) {
    char path[512]; snprintf(path, sizeof path, "%s/ops_%s.bin", dir, name);
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

static int compare(const char *name, const float *got, const float *want, int n, float tol) {
    float worst = 0.f; int bad = 0;
    for (int i = 0; i < n; i++) {
        float e = fabsf(got[i] - want[i]);
        if (e > worst) worst = e;
        if (!(e <= tol + tol * fabsf(want[i]))) bad++;
    }
    printf("%s  %-10s max err %.2e (%d/%d off)\n", bad ? "FAIL" : "PASS", name, worst, bad, n);
    return bad != 0;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/fixtures";
    int failed = 0; Fix f; float *out = malloc(sizeof(float) * 8192);
    if (load(dir, "rmsnorm", &f) == 0) { op_rmsnorm(out, f.a[0], f.a[1], f.len[0], 1e-6f);
        failed |= compare("rmsnorm", out, f.a[2], f.len[0], TOL); drop(&f); } else failed = 1;
    if (load(dir, "silu_mul", &f) == 0) { op_silu_mul(out, f.a[0], f.a[1], f.len[0]);
        failed |= compare("silu_mul", out, f.a[2], f.len[0], TOL); drop(&f); } else failed = 1;
    if (load(dir, "sigmoid", &f) == 0) { op_sigmoid(out, f.a[0], f.len[0]);
        failed |= compare("sigmoid", out, f.a[1], f.len[0], TOL); drop(&f); } else failed = 1;
    if (load(dir, "softmax", &f) == 0) { op_softmax(out, f.a[0], f.len[0]);
        failed |= compare("softmax", out, f.a[1], f.len[0], 1e-6f); drop(&f); } else failed = 1;
    if (load(dir, "topk", &f) == 0) {
        int k = (int)f.a[1][0], idx[64]; float got[64];
        op_topk(idx, f.a[0], f.len[0], k);
        for (int i = 0; i < k; i++) got[i] = (float)idx[i];
        failed |= compare("topk", got, f.a[2], k, 0.f); drop(&f);
    } else failed = 1;
    {   /* topk edge cases: ties keep the lower index first; nothing written past idx[k-1] */
        const float x[10] = { 1.f, 5.f, 3.f, 5.f, 2.f, 5.f, 0.f, 4.f, -1.f, 3.f };
        const int want[4] = { 1, 3, 5, 7 };
        int idx[5] = { -7, -7, -7, -7, -7 };
        op_topk(idx, x, 10, 4);
        int ok = idx[4] == -7;
        for (int i = 0; i < 4; i++) ok &= idx[i] == want[i];
        printf("%s  topk-edge  ties + no write past k (got %d %d %d %d, guard %d)\n",
               ok ? "PASS" : "FAIL", idx[0], idx[1], idx[2], idx[3], idx[4]);
        failed |= !ok;
    }
    {   /* tie with the current k-th value must not be inserted (and must not write idx[k]) */
        const float x[3] = { 5.f, 4.f, 4.f };
        int idx[3] = { -7, -7, -7 };
        op_topk(idx, x, 3, 2);
        int ok = idx[0] == 0 && idx[1] == 1 && idx[2] == -7;
        printf("%s  topk-tie-k boundary tie (got %d %d, guard %d)\n", ok ? "PASS" : "FAIL", idx[0], idx[1], idx[2]);
        failed |= !ok;
    }
    const char *ropes[] = { "rope_p0", "rope_p7", "rope_p3001" };
    for (int r = 0; r < 3; r++) {
        if (load(dir, ropes[r], &f) != 0) { failed = 1; continue; }
        const float *p = f.a[0];
        memcpy(out, f.a[1], sizeof(float) * f.len[1]);
        op_rope(out, (int)p[0], (int)p[1], (int)p[2], (int)p[3], p[4]);
        failed |= compare(ropes[r], out, f.a[2], f.len[1], 2e-5f); drop(&f);
    }
    free(out);
    printf(failed ? "OPS TESTS FAILED\n" : "ALL OPS TESTS PASSED\n");
    return failed;
}
