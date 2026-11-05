#include <stdint.h>
int encode_utf8_scalar(uint32_t cp, uint8_t out[4]) {
    if (!out || cp > 0x10ffffu || (cp >= 0xd800u && cp <= 0xdfffu)) return -1;
    if (cp <= 0x7fu) { out[0] = (uint8_t)cp; return 1; }
    if (cp <= 0x7ffu) {
        out[0] = (uint8_t)(0xc0u | (cp >> 6)); out[1] = (uint8_t)(0x80u | (cp & 0x3fu)); return 2;
    }
    if (cp <= 0xffffu) {
        out[0] = (uint8_t)(0xe0u | (cp >> 12)); out[1] = (uint8_t)(0x80u | ((cp >> 6) & 0x3fu));
        out[2] = (uint8_t)(0x80u | (cp & 0x3fu)); return 3;
    }
    out[0] = (uint8_t)(0xf0u | (cp >> 18)); out[1] = (uint8_t)(0x80u | ((cp >> 12) & 0x3fu));
    out[2] = (uint8_t)(0x80u | ((cp >> 6) & 0x3fu)); out[3] = (uint8_t)(0x80u | (cp & 0x3fu));
    return 4;
}
