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
