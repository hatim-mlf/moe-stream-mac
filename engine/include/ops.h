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
