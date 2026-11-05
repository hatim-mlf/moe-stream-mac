#include <limits.h>
#include <stddef.h>
#include <stdint.h>
int hex_encode_lower(const uint8_t *src, size_t n, char *dst, size_t cap) {
    static const char hex[] = "0123456789abcdef";
    if (!dst || (n && !src) || n > ((size_t)INT_MAX / 2u)) return -1;
    size_t chars = n * 2u;
    if (cap < chars + 1u) return -1;
    for (size_t i = 0; i < n; ++i) {
        dst[2u*i] = hex[src[i] >> 4];
        dst[2u*i+1u] = hex[src[i] & 15u];
    }
    dst[chars] = '\0';
    return (int)chars;
}
