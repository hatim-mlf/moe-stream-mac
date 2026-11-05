#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
int16_t a[]={3,-2,0,100};assert(sum_i16_before_zero(a,4)==1);
int16_t b[]={0,7};assert(sum_i16_before_zero(b,2)==0);
assert(sum_i16_before_zero(NULL,0)==0);
return 0;
}
