#include <stdint.h>
int32_t lerp_i32_u16(int32_t a, int32_t b, uint16_t weight) {
    const int64_t den = 65535;
    int64_t num = ((int64_t)b - (int64_t)a) * (int64_t)weight;
    int64_t q = num / den, r = num % den;
    int64_t mag = r < 0 ? -r : r;
    if (mag * 2 > den || (mag * 2 == den && (q % 2 != 0))) q += num < 0 ? -1 : 1;
    return (int32_t)((int64_t)a + q);
}
