#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
assert(sat_sub_i16(7,3)==4); assert(sat_sub_i16(-7,3)==-10);
assert(sat_sub_i16(INT16_MAX,-1)==INT16_MAX);
assert(sat_sub_i16(INT16_MIN,1)==INT16_MIN);
return 0;
}
