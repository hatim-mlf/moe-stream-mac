# Task: implement engine/src/qmat.c

Write `engine/src/qmat.c`, implementing the two functions declared in this header (`#include "qmat.h"`):

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

#endif
```

## Requirements

- Portable C99, only `<stdint.h>`, `<string.h>`, `<stddef.h>` and `"qmat.h"`. No allocation, no globals.
- `bf16_to_f32`: shift the 16 bits into the top half of a uint32, then `memcpy` it into a float (no pointer casts).
- `qmat_matvec`:
  - Validate first: `bits` must be 4 or 8; `group_size` must be 32, 64 or 128; `cols` must be > 0 and a
    multiple of `group_size`; `rows` must be > 0. On invalid input return -1 **without writing to y**.
  - For each row, for each group: accumulate `sum_q = Σ q*x` and `sum_x = Σ x` over the group, then add
    `scale * sum_q + bias * sum_x` to the row total (this is exact algebra and avoids dequantising each weight).
  - Accumulate in `float`. Unpack values from each uint32 with shifts and a mask `(1u << bits) - 1`,
    least-significant value first, exactly as the header describes.
  - Row stride in `w` is `cols * bits / 32` uint32s; row stride in scales/biases is `cols / group_size`.
- Keep functions under 50 lines; use small `static` helpers if needed. Brief comments only where non-obvious.

Return only the complete `qmat.c` in one fenced code block.
