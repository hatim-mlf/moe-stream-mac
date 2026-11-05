#include <stddef.h>
int copy_matrix_row_f32(const float*m,size_t rows,size_t cols,size_t row,float*out,size_t cap){if(row>rows||cap<cols||(cols&&(!m||!out)))return -1;for(size_t c=0;c<cols;++c)out[c]=m[row*cols+c];return 0;}
