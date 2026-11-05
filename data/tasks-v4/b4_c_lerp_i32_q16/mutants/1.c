#include <stdint.h>
int32_t lerp_i32_u16(int32_t a,int32_t b,uint16_t w){int64_t n=((int64_t)b-a)*w;return (int32_t)((int64_t)a+(n+32767)/65535);}
