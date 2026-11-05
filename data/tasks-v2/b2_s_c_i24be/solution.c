#include <stddef.h>
#include <stdint.h>
#include <limits.h>
int decode_i24_be(const uint8_t *data, size_t len, size_t offset, int32_t *out) {
    if (!data || !out || offset > len || len - offset < 3) return -1;
    uint32_t u = ((uint32_t)data[offset] << 16) | ((uint32_t)data[offset+1] << 8) | data[offset+2];
    int32_t value = (u & UINT32_C(0x800000)) ? (int32_t)(u | UINT32_C(0xFF000000)) : (int32_t)u;
    *out = value;
    return 0;
}
