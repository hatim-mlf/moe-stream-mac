#include <stddef.h>
#include <stdint.h>
int64_t max_subarray_sum_i32(const int32_t *values,size_t n){
 if(n==0)return 0;
 int64_t ending=values[0],best=ending;
 for(size_t i=1;i<n;i++){
  int64_t v=values[i];
  ending=ending>0?ending+v:v;
  if(ending>best)best=ending;
 }
 return best;
}
