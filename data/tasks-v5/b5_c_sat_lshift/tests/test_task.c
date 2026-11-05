#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
    assert(sat_lshift_i32(3,2)==12 && sat_lshift_i32(-3,2)==-12);
    assert(sat_lshift_i32(7,0)==7);
    assert(sat_lshift_i32(INT32_MAX,1)==INT32_MAX);
    assert(sat_lshift_i32(INT32_MIN,1)==INT32_MIN);
    assert(sat_lshift_i32(-1,31)==INT32_MIN); return 0;
}
