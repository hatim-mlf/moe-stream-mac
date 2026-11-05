#include <stdint.h>
#include <math.h>
int q8_8_encode(float x, int16_t *out) {
    if (!out || !isfinite(x) || x*256.0f < -32768.0f || x*256.0f > 32767.0f) return -1;
    *out=(int16_t)(x*256.0f);
    return 0;
}
