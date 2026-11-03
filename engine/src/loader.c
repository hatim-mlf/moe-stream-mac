#include "loader.h"
#include <stdio.h>
#include <stdlib.h>

static void *read_raw(const K3St *s, const char *name, const K3Tensor **t, K3Dtype want) {
    *t = k3_st_find(s, name);
    if (!*t) { k3_st_note_missing(s, name); fprintf(stderr, "loader: missing %s\n", name); return NULL; }
    if ((*t)->dtype != want) { fprintf(stderr, "loader: %s has unexpected dtype %d\n", name, (*t)->dtype); return NULL; }
    void *buf = malloc((size_t)(*t)->nbytes);
    if (!buf || k3_st_read(s, *t, buf) != (*t)->nbytes) {
        fprintf(stderr, "loader: cannot read %s\n", name); free(buf); return NULL;
    }
    return buf;
}

int load_qmat(const K3St *s, const char *base, int group_size, QMat *out, int *n_experts) {
    char name[512];
    const K3Tensor *tw, *ts, *tb;
    snprintf(name, sizeof name, "%s.weight", base); uint32_t *w = read_raw(s, name, &tw, K3_DT_U32);
    snprintf(name, sizeof name, "%s.scales", base); uint16_t *sc = read_raw(s, name, &ts, K3_DT_BF16);
    snprintf(name, sizeof name, "%s.biases", base); uint16_t *bi = read_raw(s, name, &tb, K3_DT_BF16);
    if (!w || !sc || !bi) { free(w); free(sc); free(bi); return -1; }
    const int d = tw->ndim;
    const int rows = (int)tw->shape[d - 2], packed = (int)tw->shape[d - 1];
    const int cols = (int)ts->shape[d - 1] * group_size, bits = packed * 32 / cols;
    if (packed * 32 != bits * cols || (bits != 4 && bits != 8)) {
        fprintf(stderr, "loader: %s: cannot derive bits (packed %d, cols %d)\n", base, packed, cols);
        free(w); free(sc); free(bi); return -1;
    }
    *out = (QMat){ w, sc, bi, rows, cols, bits, group_size };
    if (n_experts) *n_experts = d == 3 ? (int)tw->shape[0] : 0;
    return 0;
}

QMat qmat_expert(const QMat *all, int e) {
    QMat m = *all;
    m.w += (size_t)e * all->rows * (all->cols * all->bits / 32);
    m.scales += (size_t)e * all->rows * (all->cols / all->group_size);
    m.biases += (size_t)e * all->rows * (all->cols / all->group_size);
    return m;
}

float *load_f32(const K3St *s, const char *name, int64_t *n) {
    const K3Tensor *t = k3_st_find(s, name);
    if (!t) { k3_st_note_missing(s, name); fprintf(stderr, "loader: missing %s\n", name); return NULL; }
    int64_t count = 1;
    for (int i = 0; i < t->ndim; i++) count *= t->shape[i];
    float *buf = malloc(sizeof(float) * (size_t)count);
    if (!buf || k3_st_read_f32(s, t, buf) != count) { fprintf(stderr, "loader: cannot read %s\n", name); free(buf); return NULL; }
    if (n) *n = count;
    return buf;
}

void free_qmat(QMat *m) {
    free((void *)m->w); free((void *)m->scales); free((void *)m->biases);
    *m = (QMat){ 0 };
}
