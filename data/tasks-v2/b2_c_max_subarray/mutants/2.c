#include <stddef.h>
#include <stdint.h>
int64_t max_subarray_sum_i32(const int32_t*x,size_t n){if(!n)return 0;int64_t cur=x[1],best=cur;for(size_t i=2;i<n;i++){cur=cur>0?cur+x[i]:x[i];if(cur>best)best=cur;}return best;}
