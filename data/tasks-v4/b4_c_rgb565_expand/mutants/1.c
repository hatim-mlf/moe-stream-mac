#include <stdint.h>
void rgb565_expand(uint16_t pixel, uint8_t *r, uint8_t *g, uint8_t *b) {
    *r = (uint8_t)(((pixel >> 11) & 31u) << 3);
    *g = (uint8_t)(((pixel >> 5) & 63u) << 2);
    *b = (uint8_t)((pixel & 31u) << 3);
}
