#include <stddef.h>
float trace_rowmajor_f32(const float *matrix,size_t rows,size_t cols) {
    size_t n=rows<cols?rows:cols; float sum=0.0f;
    for(size_t i=0;i<n;i++) sum+=matrix[i*cols+i];
    return sum;
}
