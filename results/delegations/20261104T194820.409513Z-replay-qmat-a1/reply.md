```c
#include "qmat.h"
#include <string.h>
#include <stddef.h>

float bf16_to_f32(uint16_t h) {
    uint32_t v = (uint32_t)h << 16;
    float f;
    memcpy(&f, &v, sizeof(f));
    return f;
}

int qmat_matvec(const QMat *m, const float *x, float *y) {
    if (m->bits != 4 && m->bits != 8) return -1;
    if (m->group_size != 32 && m->group_size != 64 && m->group_size != 128) return -1;
    if (m->cols <= 0 || m->rows <= 0) return -1;
    if (m->cols % m->group_size != 0) return -1;

    const uint32_t *w = m->w;
    const uint16_t *scales = m->scales;
    const uint16_t *biases = m->biases;
    int rows = m->rows;
    int cols = m->cols;
    int bits = m->bits;
    int group_size = m->group_size;
    int w_stride = cols * bits / 32;
    int s_stride = cols / group_size;
    uint32_t mask = (1u << bits) - 1;

    for (int r = 0; r < rows; r++) {
        float row_sum = 0.0f;
        const uint32_t *w_row = w + r * w_stride;
        const uint16_t *s_row = scales + r * s_stride;
        const uint16_t *b_row = biases + r * s_stride;

        for (int g = 0; g < cols; g += group_size) {
            float sum_q = 0.0f;
            float sum_x = 0.0f;
            float scale = bf16_to_f32(s_row[g / group_size]);
            float bias = bf16_to_f32(b_row[g / group_size]);

            for (int c = 0; c < group_size; c++) {
                int idx = g + c;
                int word_idx = idx * bits / 32;
                int bit_offset = idx * bits % 32;
                uint32_t val = (w_row[word_idx] >> bit_offset) & mask;
                sum_q += (float)val * x[idx];
                sum_x += x[idx];
            }
            row_sum += scale * sum_q + bias * sum_x;
        }
        y[r] = row_sum;
    }
    return 0;
}
```