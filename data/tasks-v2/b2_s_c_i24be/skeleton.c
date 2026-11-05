#include <stddef.h>
#include <stdint.h>
int decode_i24_be(const uint8_t *data, size_t len, size_t offset, int32_t *out) {
    /* TODO: validate the slice, decode big-endian signed 24-bit, and store atomically. */
}
