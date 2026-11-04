#include "qmat.h"
#include <string.h>
#include <stddef.h>

float bf16_to_f32(uint16_t h) {
    uint32_t v = (uint32_t)h << 16;
    float f;
    memcpy(&f, &v, sizeof f);
    return f;
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
    uint32_t mask = (1u << bits) - 1u;

    /* Rows are independent: split them across threads (OMP_NUM_THREADS). Tiny matrices stay serial. */
    #pragma omp parallel for schedule(static) if (rows >= 64)
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const uint32_t *wp = m->w + r * stride_w;
        const uint16_t *sp = m->scales + r * stride_s;
        const uint16_t *bp = m->biases + r * stride_s;

        for (int g = 0; g < cols; g += group_size) {
            float sum_q = 0.0f;
            float sum_x = 0.0f;
            int remaining = group_size;

            while (remaining > 0) {
                uint32_t word = *wp++;
                int n = remaining < 32 / bits ? remaining : 32 / bits;
                for (int i = 0; i < n; i++) {
                    uint32_t val = (word >> (i * bits)) & mask;
                    sum_q += (float)val * x[g + (group_size - remaining) + i];
                    sum_x += x[g + (group_size - remaining) + i];
                }
                remaining -= n;
            }

            float scale = bf16_to_f32(sp[0]);
            float bias = bf16_to_f32(bp[0]);
            sum += scale * sum_q + bias * sum_x;
            sp++;
            bp++;
        }
        y[r] = sum;
    }

    return 0;
}
