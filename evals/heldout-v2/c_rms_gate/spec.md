Target: candidate.c (C99)

Apply residual RMS normalization followed by an elementwise gate.

Exact signature:
```c
int rms_gate_vector(const float *x, const float *residual, const float *gate, int n, float eps, float *out);
```

Let `z[i] = x[i] + residual[i]`. Compute `denom = sqrt(sum(z[i]*z[i])/n + eps)`, then `out[i] = z[i]/denom * gate[i]`. Require non-null pointers, `n > 0`, finite positive eps, and finite z and gate values. Return 0 or -1; invalid input leaves output unchanged. Inputs and output are disjoint. Use float accumulation, no allocation.

Return only the complete file in one fenced c block.
