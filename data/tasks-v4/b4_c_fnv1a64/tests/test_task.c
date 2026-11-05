#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    const uint8_t a[] = {0, 1, 255};
    assert(fnv1a64_update(7, NULL, 0) == 7);
    assert(fnv1a64_update(0, a, 3) == UINT64_C(0x0002ff0000023424));
    assert(fnv1a64_update(5, a, 2) == fnv1a64_update(fnv1a64_update(5, a, 1), a+1, 1));
    return 0;
}
