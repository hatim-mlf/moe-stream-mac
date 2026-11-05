#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
uint8_t a[]={0,4,0,255,0}; assert(count_zero_u8(a,5)==3);
assert(count_zero_u8(NULL,0)==0); uint8_t b[]={1,2};assert(count_zero_u8(b,2)==0);
return 0;
}
