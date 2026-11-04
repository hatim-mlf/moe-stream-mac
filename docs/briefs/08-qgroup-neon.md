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
