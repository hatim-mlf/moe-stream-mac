#include <stdint.h>
#include <math.h>
int quantize_sqrt_u8(float x,uint8_t*out){if(!out||!isfinite(x)||x<0.0f||x>1.0f)return -1;unsigned q=(unsigned)floor(sqrt((double)x)*255.0+0.5);*out=(uint8_t)q;return 0;}
