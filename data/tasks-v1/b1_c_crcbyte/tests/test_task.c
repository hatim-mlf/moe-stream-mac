#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    const uint8_t s[]="123456789";
    assert((crc32_ieee_update(UINT32_C(0xffffffff),s,9)^UINT32_C(0xffffffff))==UINT32_C(0xcbf43926));
    assert(crc32_ieee_update(42,s,0)==42);
}
