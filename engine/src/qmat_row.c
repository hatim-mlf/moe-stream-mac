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
    const int per_word = 32 / m->bits;
    const uint32_t mask = (1u << m->bits) - 1;

    for (int c = 0; c < m->cols; c++) {
        const uint32_t q = (row_w[c / per_word] >> ((c % per_word) * m->bits)) & mask;
        const int g = c / m->group_size;
        out[c] = (float)q * bf16_to_f32(row_s[g]) + bf16_to_f32(row_b[g]);
    }
    return 0;
}
