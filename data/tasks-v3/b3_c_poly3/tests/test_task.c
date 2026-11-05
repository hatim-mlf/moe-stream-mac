#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
float c[]={1,2,3,4}; assert(cubic_horner_f32(0,c)==1);
assert(cubic_horner_f32(1,c)==10); assert(cubic_horner_f32(-1,c)==-2);
return 0;
}
