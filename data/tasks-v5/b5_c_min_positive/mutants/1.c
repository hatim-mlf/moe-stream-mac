#include <stddef.h>
int min_positive_f32(const float*x,size_t n,float*out){if(!out||(n&&!x))return -1;int f=0;float b=0;for(size_t i=0;i<n;++i)if(x[i]>=0&&(!f||x[i]<b)){b=x[i];f=1;}if(!f)return -1;*out=b;return 0;}
