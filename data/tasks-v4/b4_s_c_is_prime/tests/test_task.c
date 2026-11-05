#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    assert(is_prime_u32(0) == 0 && is_prime_u32(1) == 0);
    assert(is_prime_u32(2) == 1 && is_prime_u32(97) == 1);
    assert(is_prime_u32(9) == 0 && is_prime_u32(100) == 0);
    return 0;
}
