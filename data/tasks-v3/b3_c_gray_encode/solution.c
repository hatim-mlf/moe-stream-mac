#include <stdint.h>
uint32_t gray_encode_u32(uint32_t value) { return value ^ (value >> 1); }
