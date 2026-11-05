#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include "../candidate.c"
int main(void) {
 int32_t a[]={10,3,-2,4}; assert(alternating_sum_i32(a,4)==1);
 int32_t b[]={INT32_MIN}; assert(alternating_sum_i32(b,1)==INT32_MIN);
 assert(alternating_sum_i32(NULL,0)==0); return 0;
}
