#include <stdint.h>
#include <limits.h>
int div_round_even_i64(int64_t n,int64_t d,int64_t*out){if(!out||!d||(n==INT64_MIN&&d==-1))return -1;int64_t q=n/d,r=n%d;if(r){uint64_t ar=r<0?(uint64_t)(-r):(uint64_t)r,ad=d<0?(uint64_t)(-d):(uint64_t)d;if(ar>=ad-ar)q+=((n<0)==(d<0))?1:-1;}*out=q;return 0;}
