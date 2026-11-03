# Task: implement engine/src/ops.c

Write `engine/src/ops.c`, implementing every function declared in this header (`#include "ops.h"`). Follow each comment exactly: it is the spec.

```c
/* ops.h: elementwise and small vector ops for Qwen3.5 / Qwen3.6 (qwen3_5_moe). All float32.
 * out and inputs may NOT alias unless stated. n > 0 always.
 */
#ifndef OPS_H
#define OPS_H

/* out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]      (mean over the n values; accumulate in double) */
void op_rmsnorm(float *out, const float *x, const float *w, int n, float eps);

/* silu(v) = v / (1 + exp(-v)).  out[i] = silu(gate[i]) * up[i] */
void op_silu_mul(float *out, const float *gate, const float *up, int n);

/* out[i] = 1 / (1 + exp(-x[i])) */
void op_sigmoid(float *out, const float *x, int n);

/* Numerically stable softmax: subtract max(x) first, accumulate the sum in double. */
void op_softmax(float *out, const float *x, int n);

/* Indices of the k largest x values, written to idx[0..k) in DESCENDING order of value
 * (ties: lower index first). 1 <= k <= n <= 4096. x is not modified. */
void op_topk(int *idx, const float *x, int n, int k);

/* Rotary embedding, "rotate half" (non-interleaved) form, IN PLACE on x = n_heads consecutive
 * heads of head_dim floats. Only the first rot_dims of each head rotate; the rest are untouched.
 * For head h, i in [0, rot_dims/2):
 *   inv_freq = theta ^ (-2i / rot_dims);   a = pos * inv_freq   (compute a in double)
 *   x1 = x[h*head_dim + i];  x2 = x[h*head_dim + i + rot_dims/2]
 *   x[h*head_dim + i]              = x1*cos(a) - x2*sin(a)
 *   x[h*head_dim + i + rot_dims/2] = x2*cos(a) + x1*sin(a)
 * rot_dims is even and <= head_dim. */
void op_rope(float *x, int n_heads, int head_dim, int rot_dims, int pos, float theta);

#endif
```

## Requirements

- Portable C99. Only `<math.h>`, `<string.h>` and `"ops.h"`. No allocation, no globals, no static buffers.
- Use `expf`/`sqrt`/`exp` from `<math.h>`. Where the spec says double, use double.
- `op_topk`: n ≤ 4096 and k ≤ 64. Use simple selection: keep a sorted list of the best k so far, inserting
  each element (strictly greater moves ahead; equal values keep the earlier index first).
- `op_rope` works in place: read x1 and x2 into locals before writing either.
- Functions under 50 lines. Brief comments only where non-obvious.

Return only the complete `ops.c` in one fenced code block.
