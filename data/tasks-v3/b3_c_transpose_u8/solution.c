#include <stdint.h>
#include <stddef.h>
void transpose_u8(const uint8_t *src, size_t rows, size_t cols, uint8_t *dst) {
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            dst[c * rows + r] = src[r * cols + c];
}
