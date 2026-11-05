#include <stddef.h>
#include <stdint.h>
int64_t max_subarray_sum_i32(const int32_t*x,size_t n){if(!n)return 0;int64_t cur=0,best=0;for(size_t i=0;i<n;i++){cur+=x[i];if(cur<0)cur=0;if(cur>best)best=cur;}return best;}
