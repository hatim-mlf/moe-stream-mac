#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    uint32_t out = 99;
    assert(lcm_u32(21, 6, &out) == 0 && out == 42);
    assert(lcm_u32(0, UINT32_MAX, &out) == 0 && out == 0);
    out = 17; assert(lcm_u32(UINT32_MAX, UINT32_MAX-1u, &out) == -1 && out == 17);
    return 0;
}
