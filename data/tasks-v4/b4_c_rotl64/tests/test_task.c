#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    const uint64_t x=UINT64_C(0x0123456789abcdef);
    assert(rotate_left_u64(x,0)==x && rotate_left_u64(x,64)==x);
    assert(rotate_left_u64(x,8)==UINT64_C(0x23456789abcdef01));
    assert(rotate_left_u64(x,63)==UINT64_C(0x8091a2b3c4d5e6f7));
    return 0;
}
