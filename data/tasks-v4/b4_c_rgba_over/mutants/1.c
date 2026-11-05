#include <stdint.h>
void rgba_over_opaque(const uint8_t fg[4], const uint8_t bg[3], uint8_t out[4]) {
    unsigned alpha = fg[3];
    for (unsigned c = 0; c < 3; ++c)
        out[c] = (uint8_t)(((unsigned)fg[c] * alpha + (unsigned)bg[c] * (255u-alpha)) / 255u);
    out[3] = 255u;
}
