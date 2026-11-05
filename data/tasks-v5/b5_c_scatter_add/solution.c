#include <stddef.h>
int scatter_add_f32(float*dst,size_t n,const size_t*idx,const float*v,size_t k){if((n&&!dst)||(k&&(!idx||!v)))return -1;for(size_t j=0;j<k;++j)if(idx[j]>=n)return -1;for(size_t j=0;j<k;++j)dst[idx[j]]+=v[j];return 0;}
