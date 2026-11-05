#include <stdint.h>
#include <limits.h>
int parse_milli_i64(const char *text, int64_t *out) {
    if (!text || !out) return -1;
    const unsigned char *p = (const unsigned char *)text;
    int negative = 0;
    if (*p == '+' || *p == '-') { negative = (*p == '-'); p++; }
    if (*p < '0' || *p > '9') return -1;
    uint64_t whole = 0;
    while (*p >= '0' && *p <= '9') {
        unsigned d = (unsigned)(*p++ - '0');
        if (whole > UINT64_MAX / 10u || whole * 10u > UINT64_MAX - d) return -1;
        whole = whole * 10u + d;
    }
    unsigned frac = 0, digits = 0;
    if (*p == '.') {
        p++;
        while (*p >= '0' && *p <= '9') {
            if (digits == 3) return -1;
            frac = frac * 10u + (unsigned)(*p++ - '0');
            digits++;
        }
        if (digits == 0) return -1;
    }
    if (*p != '\0') return -1;
    while (digits++ < 3) frac *= 10u;
    uint64_t limit = negative ? (uint64_t)INT64_MAX + 1u : (uint64_t)INT64_MAX;
    if (whole > (limit - frac) / 1000u) return -1;
    uint64_t mag = whole * 1000u + frac;
    int64_t result;
    if (negative && mag == (uint64_t)INT64_MAX + 1u) result = INT64_MIN;
    else result = negative ? -(int64_t)mag : (int64_t)mag;
    *out = result;
    return 0;
}
