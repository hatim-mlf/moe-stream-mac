#include <stdint.h>
#include <math.h>
int quantize_sqrt_u8(float x,uint8_t*out){if(!out||!isfinite(x)||x<0||x>1)return -1;*out=(uint8_t)floor(x*255.0+0.5);return 0;}
