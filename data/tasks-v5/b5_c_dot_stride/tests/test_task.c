#include <assert.h>
#include "../candidate.c"
int main(void){float a[]={1,99,2,99,3},b[]={4,99,5,99,6};assert(dot_stride_f32(a,2,b,2,3)==32);assert(dot_stride_f32(NULL,1,NULL,1,0)==0);return 0;}
