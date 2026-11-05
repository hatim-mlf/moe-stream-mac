#include <stddef.h>
void max_pool2x2_f32(const float *x,size_t r,size_t c,float *o){size_t h=(r+1)/2,w=(c+1)/2;for(size_t i=0;i<h;i++)for(size_t j=0;j<w;j++){size_t p=(2*i)*c+2*j;float m=x[p];if(x[p+1]>m)m=x[p+1];if(x[p+c]>m)m=x[p+c];if(x[p+c+1]>m)m=x[p+c+1];o[i*w+j]=m;}}
