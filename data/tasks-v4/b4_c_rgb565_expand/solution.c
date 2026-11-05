#include <stdint.h>
void rgb565_expand(uint16_t pixel, uint8_t *r, uint8_t *g, uint8_t *b) {
    uint8_t rv = (uint8_t)((pixel >> 11) & 31u);
    uint8_t gv = (uint8_t)((pixel >> 5) & 63u);
    uint8_t bv = (uint8_t)(pixel & 31u);
    *r = (uint8_t)((rv << 3) | (rv >> 2));
    *g = (uint8_t)((gv << 2) | (gv >> 4));
    *b = (uint8_t)((bv << 3) | (bv >> 2));
}
