/* test_qmat: compare qmat_matvec against MLX ground-truth fixtures, plus input validation. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qmat.h"

#define REL_TOL 1e-4f
#define ABS_TOL 1e-4f

static void *read_n(FILE *f, size_t n, size_t sz) {
    void *p = malloc(n * sz);
    if (!p || fread(p, sz, n, f) != n) { free(p); return NULL; }
    return p;
}

static int run_fixture(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("FAIL  %s: cannot open\n", path); return 1; }
    int32_t hdr[4];
    if (fread(hdr, sizeof hdr, 1, f) != 1) { fclose(f); printf("FAIL  %s: short header\n", path); return 1; }
    const int rows = hdr[0], cols = hdr[1], bits = hdr[2], gs = hdr[3];
    const size_t nw = (size_t)rows * cols * bits / 32, ng = (size_t)rows * (cols / gs);
    uint32_t *w = read_n(f, nw, 4);
    uint16_t *sc = read_n(f, ng, 2), *bi = read_n(f, ng, 2);
    float *x = read_n(f, cols, 4), *want = read_n(f, rows, 4), *got = calloc(rows, 4);
    fclose(f);
    int failed = 1;
    if (w && sc && bi && x && want && got) {
        QMat m = { w, sc, bi, rows, cols, bits, gs };
        if (qmat_matvec(&m, x, got) != 0) {
            printf("FAIL  %s: qmat_matvec returned error\n", path);
        } else {
            int bad = 0; float worst = 0.f;
            for (int r = 0; r < rows; r++) {
                float err = fabsf(got[r] - want[r]);
                if (err > worst) worst = err;
                if (err > ABS_TOL + REL_TOL * fabsf(want[r])) {
                    if (bad++ < 3) printf("      row %d: got %.6f want %.6f\n", r, got[r], want[r]);
                }
            }
            /* qmat_row: dot(row r, x) must reproduce y[r] for every row */
            float *row = malloc(sizeof(float) * cols); int row_bad = 0;
            for (int r = 0; r < rows && row; r++) {
                double d = 0; if (qmat_row(&m, r, row) != 0) { row_bad++; continue; }
                for (int c = 0; c < cols; c++) d += (double)row[c] * x[c];
                if (fabs(d - want[r]) > ABS_TOL + REL_TOL * fabsf(want[r]) * 10) row_bad++;
            }
            if (qmat_row(&m, rows, row) != -1 || qmat_row(&m, -1, row) != -1) row_bad++;
            free(row);
            if (row_bad) printf("      qmat_row: %d rows wrong or bounds not checked\n", row_bad);
            bad += row_bad;
            failed = bad != 0;
            printf("%s  %s (%dx%d, %d-bit, g%d): max abs err %.2e, %d/%d rows off\n",
                   failed ? "FAIL" : "PASS", path, rows, cols, bits, gs, worst, bad, rows);
        }
    } else {
        printf("FAIL  %s: truncated fixture\n", path);
    }
    free(w); free(sc); free(bi); free(x); free(want); free(got);
    return failed;
}

static int test_validation(void) {
    uint32_t w[8] = {0}; uint16_t s[2] = {0}, b[2] = {0}; float x[64] = {0}, y[1] = {42.f};
    const QMat bad[] = {
        { w, s, b, 1, 64, 3, 64 },   /* unsupported bits */
        { w, s, b, 1, 64, 4, 48 },   /* unsupported group size */
        { w, s, b, 1, 100, 4, 64 },  /* cols not a multiple of group size */
    };
    int failed = 0;
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        if (qmat_matvec(&bad[i], x, y) != -1 || y[0] != 42.f) {
            printf("FAIL  validation case %zu not rejected (or y modified)\n", i); failed = 1;
        }
    }
    if (bf16_to_f32(0x3F80) != 1.0f || bf16_to_f32(0xC000) != -2.0f) {
        printf("FAIL  bf16_to_f32\n"); failed = 1;
    }
    if (!failed) printf("PASS  validation + bf16\n");
    return failed;
}

int main(int argc, char **argv) {
    int failed = test_validation();
    for (int i = 1; i < argc; i++) failed |= run_fixture(argv[i]);
    printf(failed ? "QMAT TESTS FAILED\n" : "ALL QMAT TESTS PASSED\n");
    return failed;
}
