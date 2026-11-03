/* loader.h: load MLX-quantized checkpoint tensors into engine structs. Buffers are heap-allocated. */
#ifndef LOADER_H
#define LOADER_H
#include <stdint.h>
#include "k3_st.h"
#include "qmat.h"

/* Load <base>.weight/.scales/.biases. bits is derived from the shapes (cols = scales * group_size,
 * bits = packed * 32 / cols). Expert tensors ([E][rows][...]) are loaded whole; use qmat_expert().
 * Returns 0 or -1 (message on stderr). */
int load_qmat(const K3St *s, const char *base, int group_size, QMat *out, int *n_experts);

/* View of expert e inside a 3-D expert QMat loaded by load_qmat. */
QMat qmat_expert(const QMat *all, int e);

/* Load any BF16/F16/F32 tensor widened to float. Returns NULL on failure; *n gets the element count. */
float *load_f32(const K3St *s, const char *name, int64_t *n);

void free_qmat(QMat *m);
#endif
