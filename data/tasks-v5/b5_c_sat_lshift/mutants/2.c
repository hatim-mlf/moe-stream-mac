#include <stdint.h>
#include <limits.h>
int32_t sat_lshift_i32(int32_t value,unsigned shift){int64_t x=(int64_t)value*((int64_t)1<<shift);if(x>INT32_MAX)x=INT32_MAX;if(x<INT32_MIN)x=INT32_MAX;return (int32_t)x;}
