#include <stdint.h>
#include <limits.h>
int div_round_even_i64(int64_t n,int64_t d,int64_t*out){if(!out||!d||(n==INT64_MIN&&d==-1))return -1;*out=n/d;return 0;}
