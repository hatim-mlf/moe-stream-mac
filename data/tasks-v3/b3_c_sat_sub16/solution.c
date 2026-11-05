#include <stdint.h>
#include <limits.h>
int16_t sat_sub_i16(int16_t a,int16_t b) {
    int32_t value=(int32_t)a-(int32_t)b;
    if(value>INT16_MAX) value=INT16_MAX;
    if(value<INT16_MIN) value=INT16_MIN;
    return (int16_t)value;
}
