# Task: write engine/src/qmat_row.c

Implement `qmat_row` declared in this header (`#include "qmat.h"`). Reuse the same unpacking rules as the header describes.

```c
/* qmat.h: matrix-vector product with MLX affine-quantized weights (4 or 8 bit).
 *
 * STORAGE (exactly as in mlx-community safetensors):
 *   w       uint32 [rows][cols * bits / 32]  packed quantized values, row-major.
 *           Within one uint32, value j (0-based) occupies bits [j*bits, (j+1)*bits),
 *           least significant first. Values run along the column (input) dimension.
 *   scales  bf16   [rows][cols / group_size]
 *   biases  bf16   [rows][cols / group_size]
 *   Dequantised weight: W[r][c] = q[r][c] * scale[r][c / group_size] + bias[r][c / group_size]
 *
 * bf16 is the top 16 bits of an IEEE-754 float32.
 */
#ifndef QMAT_H
#define QMAT_H
#include <stdint.h>

typedef struct {
    const uint32_t *w;
    const uint16_t *scales; /* bf16 bit patterns */
    const uint16_t *biases; /* bf16 bit patterns */
    int rows;
    int cols;        /* multiple of group_size */
    int bits;        /* 4 or 8 */
    int group_size;  /* 32, 64 or 128 */
} QMat;

/* Convert a bf16 bit pattern to float. */
float bf16_to_f32(uint16_t h);

/* y[r] = sum_c W[r][c] * x[c] for r in [0, rows). x has cols floats, y has rows floats.
 * Returns 0 on success, -1 if bits/group_size/cols are invalid (y untouched). */
int qmat_matvec(const QMat *m, const float *x, float *y);

/* Dequantise row r into out[cols]: out[c] = q[r][c] * scale + bias (as above). Used for the
 * quantized embedding table. Returns 0, or -1 if r is out of range or the format is invalid. */
int qmat_row(const QMat *m, int r, float *out);

#endif
```

## Requirements

- Portable C99, `<stdint.h>` and `"qmat.h"` only. No allocation. Call `bf16_to_f32` (implemented in qmat.c).
- Validate like qmat_matvec (bits 4/8, group_size 32/64/128, cols > 0 and a multiple of group_size, rows > 0)
  AND 0 <= r < rows. Return -1 on any failure without writing to out.
- Row r starts at w + r * (cols*bits/32) and at scales/biases + r * (cols/group_size).
- One function, under 40 lines.

Return only the complete `qmat_row.c` in one fenced code block.


## Your previous attempt failed

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

Test output: every row wrong (dot(row, x) != y).

Code review (two bugs, the same two you made in qmat.c earlier):
1. Values are UNSIGNED (0..15 or 0..255). Delete the "signed 4-bit" block entirely. W = q*scale + bias with raw q.
2. Indexing is broken: `shift` resets every word so you always read row_w[0], and `g` advances every word
   instead of every group. Replace the whole loop with direct indexing:
       const int per_word = 32 / bits;
       for (int c = 0; c < cols; c++) {
           const uint32_t q = (row_w[c / per_word] >> ((c % per_word) * bits)) & mask;
           const int g = c / group_size;
           out[c] = (float)q * bf16_to_f32(row_s[g]) + bf16_to_f32(row_b[g]);
       }
   Make mask a uint32_t. Keep the validation exactly as it is now.


Fix it. Reply with the complete corrected file.