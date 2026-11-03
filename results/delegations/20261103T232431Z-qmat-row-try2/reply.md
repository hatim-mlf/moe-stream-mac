```c
#include "qmat.h"
#include <stdint.h>

int qmat_row(const QMat *m, int r, float *out) {
    if (!m || !out || !m->w || !m->scales || !m->biases ||
        (m->bits != 4 && m->bits != 8) ||
        (m->group_size != 32 && m->group_size != 64 && m->group_size != 128) ||
        m->cols <= 0 || m->cols % m->group_size != 0 ||
        m->rows <= 0 || r < 0 || r >= m->rows)
        return -1;

    const uint32_t *row_w = m->w + r * (m->cols * m->bits / 32);
    const uint16_t *row_s = m->scales + r * (m->cols / m->group_size);
    const uint16_t *row_b = m->biases + r * (m->cols / m->group_size);
    int groups = m->cols / m->group_size;
    int bits = m->bits;
    int mask = (1 << bits) - 1;
    int shift = 0;
    int g = 0;
    float scale = bf16_to_f32(row_s[0]);
    float bias = bf16_to_f32(row_b[0]);

    for (int c = 0; c < m->cols; c++) {
        uint32_t val = (row_w[shift / (32 / bits)] >> (shift % (32 / bits) * bits)) & mask;
        // Handle signed 4-bit quantization: values 0-7 are positive, 8-15 are negative (-8 to -1)
        if (bits == 4 && val >= 8) {
            val = val - 16;
        }
        out[c] = val * scale + bias;
        shift++;
        if (shift >= 32 / bits) {
            shift = 0;
            g++;
            if (g >= groups) {
                // Should not happen if cols is multiple of group_size
                break;
            }
            scale = bf16_to_f32(row_s[g]);
            bias = bf16_to_f32(row_b[g]);
        }
    }
    return 0;
}
```