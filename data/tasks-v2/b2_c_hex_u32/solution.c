#include <stdint.h>
int parse_hex_u32_exact(const char *text, uint32_t *out) {
    if (!text || !out) return -1;
    uint32_t value = 0;
    unsigned n = 0;
    for (; text[n] != '\0'; n++) {
        unsigned char c = (unsigned char)text[n];
        unsigned d;
        if (c >= '0' && c <= '9') d = (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') d = (unsigned)(c - 'a') + 10u;
        else if (c >= 'A' && c <= 'F') d = (unsigned)(c - 'A') + 10u;
        else return -1;
        if (n == 8) return -1;
        value = (value << 4) | d;
    }
    if (n == 0) return -1;
    *out = value;
    return 0;
}
