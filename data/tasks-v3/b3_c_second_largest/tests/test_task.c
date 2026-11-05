#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
int32_t x[]={5,5,3,8,8,6};int32_t out=99;
assert(second_largest_distinct_i32(x,6,&out)==0&&out==6);
int32_t y[]={-9,-4,-7};assert(second_largest_distinct_i32(y,3,&out)==0&&out==-7);
out=99;assert(second_largest_distinct_i32(x,2,&out)==-1&&out==99);
assert(second_largest_distinct_i32(NULL,0,&out)==-1&&out==99);
assert(second_largest_distinct_i32(x,3,NULL)==-1);
return 0;
}
