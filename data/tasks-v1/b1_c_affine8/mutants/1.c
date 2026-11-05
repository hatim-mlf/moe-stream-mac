#include <stdint.h>
#include <math.h>
int quantize_affine_u8(float x, float scale, int zero_point, uint8_t *out) {
    if(!out || !isfinite(x) || !isfinite(scale) || scale<=0 || zero_point<0 || zero_point>255) return -1;
    float q=x/scale+(float)zero_point;
    if(q<0) q=0; if(q>255) q=255;
    *out=(uint8_t)q; return 0;
}
