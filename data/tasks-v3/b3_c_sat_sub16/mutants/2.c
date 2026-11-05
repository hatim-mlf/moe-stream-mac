#include <stdint.h>
#include <limits.h>
int16_t sat_sub_i16(int16_t a,int16_t b){int32_t v=(int32_t)a-b;if(v>INT16_MAX)v=INT16_MAX;if(v<INT16_MIN)v=INT16_MAX;return (int16_t)v;}
