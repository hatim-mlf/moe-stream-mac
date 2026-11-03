/* real_matvec: run qmat_matvec on one quantized linear from a real MLX checkpoint.
 *   real_matvec <model dir> <tensor base name> <bits> <group size> <expert or -1> <out.f32>
 * x[c] = sin(0.37 c + 1); writes y as raw float32. tests/check_real.py compares it with MLX. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "k3_st.h"
#include "qmat.h"

static void *load(const K3St *s, const char *base, const char *suffix, const K3Tensor **out) {
    char name[512];
    snprintf(name, sizeof name, "%s.%s", base, suffix);
    const K3Tensor *t = k3_st_find(s, name);
    if (!t) { k3_st_note_missing(s, name); return NULL; }
    void *buf = malloc((size_t)t->nbytes);
    if (buf && k3_st_read(s, t, buf) != t->nbytes) { free(buf); buf = NULL; }
    *out = t;
    return buf;
}

int main(int argc, char **argv) {
    if (argc != 7) { fprintf(stderr, "usage: %s dir base bits gs expert out\n", argv[0]); return 2; }
    const int bits = atoi(argv[3]), gs = atoi(argv[4]), expert = atoi(argv[5]);
    K3St s;
    if (k3_st_open(&s, argv[1]) != 0) return 1;
    const K3Tensor *tw, *ts, *tb;
    uint32_t *w = load(&s, argv[2], "weight", &tw);
    uint16_t *sc = load(&s, argv[2], "scales", &ts), *bi = load(&s, argv[2], "biases", &tb);
    if (!w || !sc || !bi || tw->dtype != K3_DT_U32 || ts->dtype != K3_DT_BF16) {
        fprintf(stderr, "bad or missing tensors for %s\n", argv[2]); return 1;
    }
    const int e = expert < 0 ? 0 : expert, d = tw->ndim;
    const int rows = (int)tw->shape[d - 2], packed = (int)tw->shape[d - 1], cols = packed * 32 / bits;
    const size_t wstride = (size_t)rows * packed, gstride = (size_t)rows * (cols / gs);
    QMat m = { w + e * wstride, sc + e * gstride, bi + e * gstride, rows, cols, bits, gs };
    float *x = malloc(sizeof(float) * cols), *y = malloc(sizeof(float) * rows);
    for (int c = 0; c < cols; c++) x[c] = sinf(0.37f * c + 1.f);
    if (qmat_matvec(&m, x, y) != 0) { fprintf(stderr, "qmat_matvec rejected shape\n"); return 1; }
    FILE *f = fopen(argv[6], "wb");
    if (!f || fwrite(y, sizeof(float), rows, f) != (size_t)rows) return 1;
    fclose(f);
    printf("%s%s: %dx%d %d-bit -> %s\n", argv[2], expert < 0 ? "" : " (expert)", rows, cols, bits, argv[6]);
    k3_st_close(&s);
    return 0;
}
