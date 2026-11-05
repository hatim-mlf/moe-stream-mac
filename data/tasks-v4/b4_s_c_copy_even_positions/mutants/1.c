#include <limits.h>
#include <stddef.h>
#include <stdint.h>
int copy_even_positions_i32(const int32_t *s,size_t n,int32_t *d,size_t c){size_t k=n/2+n%2;if(k>(size_t)INT_MAX||c<k||(n&&!s)||(k&&!d))return -1;for(size_t i=0;i<k;i++)d[i]=s[2*i+1];return (int)k;}
