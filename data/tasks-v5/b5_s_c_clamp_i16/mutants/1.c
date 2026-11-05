#include <stdint.h>
int16_t clamp_i16_value(int16_t value,int16_t lower,int16_t upper){if(value<lower)return (int16_t)(lower+1);if(value>upper)return (int16_t)(upper-1);return value;}
