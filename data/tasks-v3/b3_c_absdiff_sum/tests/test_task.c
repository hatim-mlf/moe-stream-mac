#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
uint16_t a[]={0,65535,20,7},b[]={65535,0,12,7};
assert(sum_absdiff_u16(a,b,4)==131078u); assert(sum_absdiff_u16(NULL,NULL,0)==0);
return 0;
}
