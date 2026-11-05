#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){int32_t a[]={-2,3,-1,4,-8},b[]={-9,-2,-7},c[]={10,-20,3};
 assert(max_subarray_sum_i32(a,5)==6);
 assert(max_subarray_sum_i32(b,3)==-2);
 assert(max_subarray_sum_i32(c,3)==10);
 assert(max_subarray_sum_i32(NULL,0)==0);
 return 0;
}
