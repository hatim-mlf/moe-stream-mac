#include <assert.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
    assert(median5_i32(9,1,8,3,4)==4);
    assert(median5_i32(2,2,2,1,9)==2);
    assert(median5_i32(INT_MIN,0,INT_MAX,-1,1)==0);
}
