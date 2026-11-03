```c
#include "qmat.h"
#include <string.h>
#include <stddef.h>

float bf16_to_f32(uint16_t h) {
    uint32_t v = (uint32_t)h << 16;
    float f;
    memcpy(&f, &v, sizeof f);
    return f;
}

/* Helper: unpack 'n' values of 'bits' width from a uint32, LSB first. */
static void unpack_values(const uint32_t *wp, int n, int bits, int *vals) {
    uint32_t mask = (1u << bits) - 1u;
    for (int i = 0; i < n; i++) {
        vals[i] = (int)((*wp >> (i * bits)) & mask);
        /* Sign-extend if needed (4-bit signed: 0-7 maps to 0-7, but we need signed) */
        if (bits == 4 && vals[i] >= 8) {
            vals[i] -= 16;
        }
    }
}

int qmat_matvec(const QMat *m, const float *x, float *y) {
    if (!m || !x || !y) return -1;
    if (m->bits != 4 && m->bits != 8) return -1;
    if (m->group_size != 32 && m->group_size != 64 && m->group_size != 128) return -1;
    if (m->rows <= 0 || m->cols <= 0) return -1;
    if (m->cols % m->group_size != 0) return -1;

    int cols = m->cols;
    int rows = m->rows;
    int bits = m->bits;
    int group_size = m->group_size;
    int stride_w = cols * bits / 32;
    int stride_s = cols / group_size;
    int *vals = (int *)malloc(group_size * sizeof(int));
    if (!vals) return -1;

    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const uint32_t *wp = m->w + r * stride_w;
        const uint16_t *sp = m->scales + r * stride_s;
        const uint16_t *bp = m->biases + r * stride_s;

        for (int g = 0; g < cols; g += group_size) {
            int n = group_size;
            unpack_values(wp, n, bits, vals);
            wp++;

            float sum_q = 0.0f;
            float sum_x = 0.0f;
            for (int i = 0; i < n; i++) {
                sum_q += vals[i] * x[g + i];
                sum_x += x[g + i];
            }

            float scale = bf16_to_f32(sp[0]);
            float bias = bf16_to_f32(bp[0]);
            sum += scale * sum_q + bias * sum_x;
        }
        y[r] = sum;
    }

    free(vals);
    return 0;
}
```