#include <stdint.h>
int32_t sat_add_i24(int32_t a,int32_t b){int64_t s=(int64_t)a+b;if(s>8388607)s=8388607;if(s< -8388607)s= -8388607;return (int32_t)s;}
