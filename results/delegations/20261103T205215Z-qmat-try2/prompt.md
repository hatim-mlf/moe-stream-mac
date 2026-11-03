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


## Your previous attempt failed

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

Compiler output:
```
cc -O2 -std=c99 -Wall -Wextra -Werror -Iinclude src/qmat.c tests/test_qmat.c -o build/test_qmat -lm
src/qmat.c:37:24: error: call to undeclared library function 'malloc' with type 'void *(unsigned long)'; ISO C99 and later do not support implicit function declarations [-Wimplicit-function-declaration]
src/qmat.c:65:5: error: call to undeclared library function 'free' with type 'void (void *)'; ISO C99 and later do not support implicit function declarations [-Wimplicit-function-declaration]
2 errors generated.
```

Reminder: the brief forbids allocation. The group can be processed in a single pass without storing unpacked values.


Fix it. Reply with the complete corrected file.