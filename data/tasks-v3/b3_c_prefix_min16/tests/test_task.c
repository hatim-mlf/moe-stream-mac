#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
int16_t a[]={4,-2,-2,7},o[4]; prefix_min_i16(a,o,4);
assert(o[0]==4&&o[1]==-2&&o[2]==-2&&o[3]==-2);
int16_t x[]={3,1,5}; prefix_min_i16(x,x,3); assert(x[0]==3&&x[1]==1&&x[2]==1);
prefix_min_i16(NULL,NULL,0);
return 0;
}
