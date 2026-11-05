#include <stdint.h>
#include <math.h>
int quantize_symmetric_s8(float x,float scale,int8_t *out) {
    if(!out || !isfinite(x) || !isfinite(scale) || scale<=0)return -1;
    float q=roundf(x/scale); if(q<-128)q=-128; if(q>127)q=127;
    *out=(int8_t)q; return 0;
}
