#include <stddef.h>
int is_printable_ascii_span(const unsigned char *data, size_t n) {
    if (n && !data) return 0;
    for (size_t i = 0; i < n; ++i) if (data[i] < 0x20u || data[i] > 0x7eu) return 0;
    return 1;
}
