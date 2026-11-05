#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
float a[]={0,-1,0,2,-3,0,-4};
assert(sign_flips_f32(a,7)==2); assert(sign_flips_f32(NULL,0)==0);
float z[]={0,0,0}; assert(sign_flips_f32(z,3)==0);
return 0;
}
