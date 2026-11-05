#include <stddef.h>
#include <stdint.h>
#include <string.h>
int copy_byte_span(const uint8_t *src, size_t src_len, size_t offset, uint8_t *dst, size_t dst_cap, size_t count) {
    if (offset > src_len || count > src_len - offset || count > dst_cap ||
        (count && (!src || !dst))) return -1;
    if (count) memmove(dst, src + offset, count);
    return 0;
}
