#include "qmat.h"
#include "qgroup.h"

#define QMAT_MAX_GROUPS 4096  /* cols / group_size; 4096 x 32 = 131072 columns */
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

    const int cols = m->cols, rows = m->rows, bits = m->bits, gs = m->group_size;
    const int ng = cols / gs, stride_w = cols * bits / 32, words_per_group = gs * bits / 32;
    if (ng > QMAT_MAX_GROUPS) return -1;

    /* sum_x of each group is the same for every row: compute it once. */
    float gsum[QMAT_MAX_GROUPS];
    for (int g = 0; g < ng; g++) {
        float s = 0.0f;
        for (int i = 0; i < gs; i++) s += x[g * gs + i];
        gsum[g] = s;
    }

    /* Rows are independent: split them across threads (OMP_NUM_THREADS). Tiny matrices stay serial. */
    #pragma omp parallel for schedule(static) if (rows >= 64)
    for (int r = 0; r < rows; r++) {
        const uint32_t *wp = m->w + (size_t)r * stride_w;
        const uint16_t *sp = m->scales + (size_t)r * ng;
        const uint16_t *bp = m->biases + (size_t)r * ng;
        float sum = 0.0f;
        for (int g = 0; g < ng; g++) {
            const float sum_q = qgroup_dot(wp + g * words_per_group, x + g * gs, gs, bits);
            sum += bf16_to_f32(sp[g]) * sum_q + bf16_to_f32(bp[g]) * gsum[g];
        }
        y[r] = sum;
    }

    return 0;
}
