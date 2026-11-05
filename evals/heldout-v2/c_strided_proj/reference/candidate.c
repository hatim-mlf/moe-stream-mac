#include <stddef.h>
int project_strided_grid(const float *a, const float *x, int rows, int cols,
                         ptrdiff_t row_stride, ptrdiff_t col_stride, ptrdiff_t x_stride, float *out) {
    int r,c;
    if (a==0 || x==0 || out==0 || rows<=0 || cols<=0 || row_stride<=0 || col_stride<=0 || x_stride<=0) return -1;
    for (r=0;r<rows;++r) { float sum=0.0f; for (c=0;c<cols;++c) sum += a[(ptrdiff_t)r*row_stride+(ptrdiff_t)c*col_stride]*x[(ptrdiff_t)c*x_stride]; out[r]=sum; }
    return 0;
}
