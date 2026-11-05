#include <stddef.h>
#include <stdint.h>
uint64_t fnv1a64_update(uint64_t state, const uint8_t *data, size_t n) {
    for (size_t i = 0; i < n; ++i) { state ^= data[i]; state *= UINT64_C(1099511628211); }
    return state;
}
