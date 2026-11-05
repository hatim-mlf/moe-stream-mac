Target: candidate.c (C99)

Compute a matrix-vector product from positive element strides, including transposed storage.

Exact signature:
```c
int project_strided_grid(const float *a, const float *x, int rows, int cols,
                         ptrdiff_t row_stride, ptrdiff_t col_stride, ptrdiff_t x_stride, float *out);
```

Read matrix element `(r,c)` at `a[r*row_stride + c*col_stride]`, vector element c at `x[c*x_stride]`, and write contiguous `out[r]` sums. Dimensions and all strides must be positive and pointers non-null. Return 0 or -1; invalid input must not change output. Output does not overlap either input. Accumulate in float; no allocation.

Return only the complete file in one fenced c block.
