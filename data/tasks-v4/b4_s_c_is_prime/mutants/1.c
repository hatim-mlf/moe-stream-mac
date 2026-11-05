#include <stdint.h>
int is_prime_u32(uint32_t n) { return n >= 2 && (n == 2 || (n & 1u)); }
