#include <stddef.h>
void matvec_rowmajor_f32(const float*m,const float*x,size_t rows,size_t cols,float*y){for(size_t r=0;r<rows;r++){float s=0;for(size_t c=0;c<cols;c++)s+=m[c*rows+r]*x[c];y[r]=s;}}
