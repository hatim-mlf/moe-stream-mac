#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
ptrdiff_t find_subspan_u8(const uint8_t *h,size_t hn,const uint8_t *n,size_t nn) {
    if((!h&&hn)||(!n&&nn)||hn>(size_t)PTRDIFF_MAX)return -1;
    if(nn==0)return 0;
    for(size_t i=0;i+nn<hn;i++) if(memcmp(h+i,n,nn)==0)return (ptrdiff_t)i;
    return -1;
}
