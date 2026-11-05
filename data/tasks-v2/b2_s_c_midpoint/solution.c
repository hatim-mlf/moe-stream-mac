#include <stdint.h>
int32_t midpoint_floor_i32(int32_t a, int32_t b) {
    int64_t sum = (int64_t)a + (int64_t)b;
    int64_t q = sum / 2;
    if (sum < 0 && sum % 2 != 0) q--;
    return (int32_t)q;
}
