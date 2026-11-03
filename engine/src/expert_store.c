#include "expert_store.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static size_t g_bytes_read;

static const K3Tensor *find(const K3St *st, const char *base, const char *suffix, K3Dtype want) {
    char name[512];
    snprintf(name, sizeof name, "%s.%s", base, suffix);
    const K3Tensor *t = k3_st_find(st, name);
    if (!t) { k3_st_note_missing(st, name); fprintf(stderr, "expert_store: missing %s\n", name); return NULL; }
    if (t->dtype != want || t->ndim != 3) { fprintf(stderr, "expert_store: %s is not a 3-D expert tensor\n", name); return NULL; }
    return t;
}

int expert_tensor_open(ExpertTensor *t, const K3St *st, const char *base, int group_size) {
    memset(t, 0, sizeof *t);
    t->st = st; t->group_size = group_size;
    if (!(t->w = find(st, base, "weight", K3_DT_U32)) || !(t->s = find(st, base, "scales", K3_DT_BF16)) ||
        !(t->b = find(st, base, "biases", K3_DT_BF16))) return -1;
    t->E = (int)t->w->shape[0]; t->rows = (int)t->w->shape[1];
    const int packed = (int)t->w->shape[2];
    t->cols = (int)t->s->shape[2] * group_size;
    t->bits = packed * 32 / t->cols;
    if (packed * 32 != t->bits * t->cols || (t->bits != 4 && t->bits != 8)) {
        fprintf(stderr, "expert_store: %s: cannot derive bits\n", base); return -1;
    }
    t->w_bytes = (size_t)t->w->nbytes / t->E;
    t->g_bytes = (size_t)t->s->nbytes / t->E;
    return 0;
}

int expert_buf_alloc(ExpertBuf *buf, const ExpertTensor *t) {
    buf->w = malloc(t->w_bytes); buf->s = malloc(t->g_bytes); buf->b = malloc(t->g_bytes);
    return buf->w && buf->s && buf->b ? 0 : -1;
}

static int pread_all(int fd, void *dst, size_t n, int64_t off) {
    char *p = dst;
    while (n > 0) {
        const ssize_t got = pread(fd, p, n, (off_t)off);
        if (got <= 0) { perror("expert_store: pread"); return -1; }
        p += got; n -= (size_t)got; off += got; g_bytes_read += (size_t)got;
    }
    return 0;
}

int expert_tensor_read(const ExpertTensor *t, int e, const ExpertBuf *buf, QMat *out) {
    if (e < 0 || e >= t->E) { fprintf(stderr, "expert_store: expert %d out of range\n", e); return -1; }
    if (pread_all(t->st->fd[t->w->shard], buf->w, t->w_bytes, t->w->off + (int64_t)e * (int64_t)t->w_bytes) ||
        pread_all(t->st->fd[t->s->shard], buf->s, t->g_bytes, t->s->off + (int64_t)e * (int64_t)t->g_bytes) ||
        pread_all(t->st->fd[t->b->shard], buf->b, t->g_bytes, t->b->off + (int64_t)e * (int64_t)t->g_bytes)) return -1;
    *out = (QMat){ buf->w, buf->s, buf->b, t->rows, t->cols, t->bits, t->group_size };
    return 0;
}

size_t expert_bytes_read(void) { return g_bytes_read; }
