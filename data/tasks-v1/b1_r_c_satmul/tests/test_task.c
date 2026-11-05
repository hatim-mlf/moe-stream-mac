#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
    assert(saturating_mul_i32(7,-6)==-42);
    assert(saturating_mul_i32(INT32_MAX,2)==INT32_MAX);
    assert(saturating_mul_i32(INT32_MIN,-1)==INT32_MAX);
    assert(saturating_mul_i32(INT32_MIN,2)==INT32_MIN);
    assert(saturating_mul_i32(0,INT32_MIN)==0);
}
