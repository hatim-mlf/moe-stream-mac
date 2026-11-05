#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
    assert(median3_i32(9,1,4)==4);
    assert(median3_i32(-2,-8,-5)==-5);
    assert(median3_i32(INT32_MIN,0,INT32_MAX)==0);
    assert(median3_i32(3,3,1)==3);
}
