/* expert_store.h: read single experts straight from the checkpoint on demand (no full load).
 * An MLX expert tensor is [E][rows][...] contiguous, so expert e is one contiguous byte range
 * in each of .weight / .scales / .biases: we pread exactly that range. */
#ifndef EXPERT_STORE_H
#define EXPERT_STORE_H
#include <stddef.h>
#include "k3_st.h"
#include "qmat.h"

typedef struct {
    const K3St *st;
    const K3Tensor *w, *s, *b;
    int E, rows, cols, bits, group_size;
    size_t w_bytes, g_bytes;          /* bytes of ONE expert in .weight and in .scales (= .biases) */
} ExpertTensor;

typedef struct { uint32_t *w; uint16_t *s, *b; } ExpertBuf;   /* room for one expert of one tensor */

/* Describe <base>.{weight,scales,biases} without reading the data. 0 or -1. */
int expert_tensor_open(ExpertTensor *t, const K3St *st, const char *base, int group_size);
/* Allocate a buffer big enough for one expert of t. 0 or -1. */
int expert_buf_alloc(ExpertBuf *buf, const ExpertTensor *t);
/* Read expert e into buf and return a QMat view of it in *out. 0 or -1. */
int expert_tensor_read(const ExpertTensor *t, int e, const ExpertBuf *buf, QMat *out);
/* Total bytes pread so far (for I/O statistics). */
size_t expert_bytes_read(void);

#endif
