#include <limits.h>
#include <stddef.h>
#include <stdint.h>
int copy_even_positions_i32(const int32_t *src, size_t n, int32_t *dst, size_t cap) {
    size_t count = n / 2u + n % 2u;
    if (count > (size_t)INT_MAX || cap < count || (n && !src) || (count && !dst)) return -1;
    for (size_t i = 0; i < count; ++i) dst[i] = src[2u*i];
    return (int)count;
}
