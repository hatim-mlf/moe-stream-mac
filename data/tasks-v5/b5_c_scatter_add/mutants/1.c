#include <stddef.h>
int scatter_add_f32(float*d,size_t n,const size_t*i,const float*v,size_t k){if((n&&!d)||(k&&(!i||!v)))return -1;for(size_t j=0;j<k;++j){if(i[j]>=n)return -1;d[i[j]]+=v[j];}return 0;}
