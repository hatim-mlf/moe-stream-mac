#include <stdint.h>
int parse_version_triplet(const char *text, uint16_t out[3]) {
    uint16_t values[3];
    const char *p = text;
    if (!p || !out || *p++ != 'v') return -1;
    for (int field = 0; field < 3; ++field) {
        if (*p < '0' || *p > '9') return -1;
        if (*p == '0' && p[1] >= '0' && p[1] <= '9') return -1;
        unsigned value = 0;
        do {
            unsigned digit = (unsigned)(*p - '0');
            if (value > (65535u - digit) / 10u) return -1;
            value = value * 10u + digit;
            ++p;
        } while (*p >= '0' && *p <= '9');
        values[field] = (uint16_t)value;
        if (field < 2) { if (*p++ != '.') return -1; }
        else if (*p != '\0') return -1;
    }
    for (int i = 0; i < 3; ++i) out[i] = values[i];
    return 0;
}
