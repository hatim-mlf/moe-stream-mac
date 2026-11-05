#include <stdint.h>
#include <math.h>
int quantize_symmetric_s8(float x,float scale,int8_t *out) {
    if (!out || !isfinite(x) || !isfinite(scale) || scale<=0.0f) return -1;
    float q=roundf(x/scale);
    if (q< -127.0f) q=-127.0f;
    if (q> 127.0f) q=127.0f;
    *out=(int8_t)q;
    return 0;
}
