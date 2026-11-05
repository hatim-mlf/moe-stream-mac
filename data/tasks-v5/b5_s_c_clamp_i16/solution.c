#include <stdint.h>
int16_t clamp_i16_value(int16_t value,int16_t lower,int16_t upper){if(value<lower)return lower;if(value>upper)return upper;return value;}
