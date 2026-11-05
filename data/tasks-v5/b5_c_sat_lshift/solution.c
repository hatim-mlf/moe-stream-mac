#include <stdint.h>
#include <limits.h>
int32_t sat_lshift_i32(int32_t value, unsigned shift) {
    int64_t result=(int64_t)value*((int64_t)1<<shift);
    if(result>INT32_MAX)result=INT32_MAX;
    if(result<INT32_MIN)result=INT32_MIN;
    return (int32_t)result;
}
