#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
int32_t x[]={4,0,-1,-5};assert(first_negative_i32(x,4)==2);
assert(first_negative_i32(x,2)==-1);assert(first_negative_i32(x,0)==-1);assert(first_negative_i32(NULL,0)==-1);
return 0;
}
