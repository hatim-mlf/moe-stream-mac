# Task: write engine/src/qgroup.c (performance kernel, ARM NEON + scalar fallback)

Implement `qgroup_dot` from this header. It is the hottest loop of the engine.

```c
/* qgroup.h: inner kernel of qmat_matvec. Dot product of one group of packed quantized values with x.
 *
 *   returns sum_{j < n} q_j * x[j]
 *   q_j is the j-th UNSIGNED value packed in w: bits = 4 or 8, value j of the group sits in
 *   w[j / (32/bits)] at bit offset (j % (32/bits)) * bits, least significant first.
 *   On a little-endian CPU this means: byte k of the group holds values 2k (low nibble) and
 *   2k+1 (high nibble) for 4-bit, and value k for 8-bit.
 *   n is 32, 64 or 128. No validation (the caller validates).
 */
#ifndef QGROUP_H
#define QGROUP_H
#include <stdint.h>

float qgroup_dot(const uint32_t *w, const float *x, int n, int bits);

#endif
```

## Requirements

- C99, `<stdint.h>`, `"qgroup.h"`, and `<arm_neon.h>` only inside `#if defined(__ARM_NEON)`.
- Two implementations in the same file, selected at compile time:
  - `#if defined(__ARM_NEON)`: NEON version (Apple M1).
  - `#else`: plain scalar version (shift + mask per value, accumulate in float).
- NEON, 4-bit: for each 16 bytes (32 values): `vld1q_u8`, `lo = vandq_u8(b, vdupq_n_u8(0x0F))` holds the EVEN values
  (j = 0,2,4,..), `hi = vshrq_n_u8(b, 4)` holds the ODD values. Load x de-interleaved with `vld2q_f32` (val[0] = even
  x, val[1] = odd x), 4 at a time. Widen u8 -> u16 (`vmovl_u8`) -> u32 (`vmovl_u16`) -> f32 (`vcvtq_f32_u32`), and
  accumulate with `vfmaq_f32` into a few float32x4_t accumulators. Reduce with `vaddvq_f32` at the end.
- NEON, 8-bit: each byte is one value in order; `vld1q_u8` 16 values, widen the same way, `vld1q_f32` x in order.
- n is 32, 64 or 128 (always a multiple of 32): no tail loop needed, but do not read past n values of x or
  n*bits/32 words of w.
- No allocation, no globals. Functions under 50 lines (static helpers per bit width are fine).

Return only the complete `qgroup.c` in one fenced code block.


## Your previous attempt failed

Compiler output for your previous attempt:
```
cc -O2 -std=c99 -Wall -Wextra -Werror -Iinclude -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include src/qgroup.c tests/test_qgroup.c -o build/test_qgroup -lm -L/opt/homebrew/opt/libomp/lib -lomp
src/qgroup.c:11:13: error: unused variable 'bit_off' [-Werror,-Wunused-variable]
src/qgroup.c:33:22: error: initializing 'uint8x16x2_t' (aka 'struct uint8x16x2_t') with an expression of incompatible type 'float32x4x2_t' (aka 'struct float32x4x2_t')
src/qgroup.c:43:38: error: passing 'float32x2_t' (vector of 2 'float32_t' values) to parameter of incompatible type 'float32x4_t' (vector of 4 'float32_t' values)
src/qgroup.c:44:38: error: passing 'float32x2_t' (vector of 2 'float32_t' values) to parameter of incompatible type 'float32x4_t' (vector of 4 'float32_t' values)
src/qgroup.c:45:38: error: passing 'float32x2_t' (vector of 2 'float32_t' values) to parameter of incompatible type 'float32x4_t' (vector of 4 'float32_t' values)
src/qgroup.c:46:38: error: passing 'float32x2_t' (vector of 2 'float32_t' values) to parameter of incompatible type 'float32x4_t' (vector of 4 'float32_t' values)
src/qgroup.c:47:38: error: passing 'float32x2_t' (vector of 2 'float32_t' values) to parameter of incompatible type 'float32x4_t' (vector of 4 'float32_t' values)
src/qgroup.c:48:38: error: passing 'float32x2_t' (vector of 2 'float32_t' values) to parameter of incompatible type 'float32x4_t' (vector of 4 'float32_t' values)
src/qgroup.c:49:38: error: passing 'float32x2_t' (vector of 2 'float32_t' values) to parameter of incompatible type 'float32x4_t' (vector of 4 'float32_t' values)
src/qgroup.c:50:38: error: passing 'float32x2_t' (vector of 2 'float32_t' values) to parameter of incompatible type 'float32x4_t' (vector of 4 'float32_t' values)
10 errors generated.
tests/test_qgroup.c:25:68: error: implicit conversion from 'int' to 'float' changes value from 2147483647 to 2147483648 [-Werror,-Wimplicit-const-int-float-conversion]
1 error generated.
```

Review: the 4-bit NEON loop is wrong beyond the type errors (it only uses 4 of the 16 lo/hi values and double-counts),
and the scalar version indexes bytes with a word index. Rewrite the file using EXACTLY this structure:

```c
#include "qgroup.h"
#if defined(__ARM_NEON)
#include <arm_neon.h>

static inline float32x4_t f4(uint32x4_t v) { return vcvtq_f32_u32(v); }

/* 4-bit: one block = 16 bytes = 32 values. lo[k] = value 2k, hi[k] = value 2k+1 (k = 0..15). */
static float dot4(const uint8_t *b8, const float *x, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
    for (int blk = 0; blk < n / 32; blk++) {
        const uint8x16_t b = vld1q_u8(b8 + 16 * blk);
        const uint8x16_t lo = vandq_u8(b, vdupq_n_u8(0x0F)), hi = vshrq_n_u8(b, 4);
        const uint16x8_t lo_a = vmovl_u8(vget_low_u8(lo)), lo_b = vmovl_high_u8(lo);   /* k 0-7, k 8-15 */
        const uint16x8_t hi_a = vmovl_u8(vget_low_u8(hi)), hi_b = vmovl_high_u8(hi);
        const float32x4_t ql[4] = { f4(vmovl_u16(vget_low_u16(lo_a))), f4(vmovl_high_u16(lo_a)),
                                    f4(vmovl_u16(vget_low_u16(lo_b))), f4(vmovl_high_u16(lo_b)) };  /* k 0-3,4-7,8-11,12-15 */
        /* TODO: qh[4] the same way from hi_a / hi_b */
        for (int h = 0; h < 4; h++) {
            const float32x4x2_t xv = vld2q_f32(x + 32 * blk + 8 * h);  /* val[0] = x[2k], val[1] = x[2k+1], k = 4h..4h+3 */
            /* TODO: acc0 = vfmaq_f32(acc0, ql[h], xv.val[0]);  acc1 = vfmaq_f32(acc1, qh[h], xv.val[1]); */
        }
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1));
}

/* 8-bit: one block = 16 bytes = 16 values in order. */
static float dot8(const uint8_t *b8, const float *x, int n) {
    /* TODO: vld1q_u8 16 values, widen to 4 float32x4_t (same widening as above), vfmaq_f32 with vld1q_f32(x + 16*blk + 4*h) */
}

float qgroup_dot(const uint32_t *w, const float *x, int n, int bits) {
    const uint8_t *b8 = (const uint8_t *)w;   /* little-endian: byte order matches value order */
    return bits == 4 ? dot4(b8, x, n) : dot8(b8, x, n);
}

#else
float qgroup_dot(const uint32_t *w, const float *x, int n, int bits) {
    const int per = 32 / bits; const uint32_t mask = (1u << bits) - 1u; float s = 0.f;
    for (int j = 0; j < n; j++) s += (float)((w[j / per] >> ((j % per) * bits)) & mask) * x[j];
    return s;
}
#endif
```
Fill in the three TODOs only. Return the whole file.


Fix it. Reply with the complete corrected file.