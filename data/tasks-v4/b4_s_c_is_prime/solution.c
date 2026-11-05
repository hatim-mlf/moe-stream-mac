#include <stdint.h>
int is_prime_u32(uint32_t value) {
    if (value < 2) return 0;
    for (uint32_t divisor = 2; divisor <= value / divisor; ++divisor)
        if (value % divisor == 0) return 0;
    return 1;
}
