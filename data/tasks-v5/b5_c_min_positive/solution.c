#include <stddef.h>
int min_positive_f32(const float*x,size_t n,float*out){if(!out||(n&&!x))return -1;int found=0;float best=0;for(size_t i=0;i<n;++i)if(x[i]>0&&(!found||x[i]<best)){best=x[i];found=1;}if(!found)return -1;*out=best;return 0;}
