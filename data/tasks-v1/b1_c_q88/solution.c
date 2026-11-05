#include <stdint.h>
#include <math.h>
#include <limits.h>
int q8_8_encode(float x, int16_t *out) {
    if (!out || !isfinite(x)) return -1;
    float q = roundf(x * 256.0f);
    if (q < (float)INT16_MIN || q > (float)INT16_MAX) return -1;
    *out = (int16_t)q;
    return 0;
}
