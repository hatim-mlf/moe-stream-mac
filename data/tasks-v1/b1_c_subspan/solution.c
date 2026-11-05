#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
ptrdiff_t find_subspan_u8(const uint8_t *haystack,size_t hay_n,const uint8_t *needle,size_t needle_n) {
    if ((!haystack && hay_n) || (!needle && needle_n) || hay_n>(size_t)PTRDIFF_MAX) return -1;
    if (needle_n==0) return 0;
    if (needle_n>hay_n) return -1;
    for (size_t i=0;i<=hay_n-needle_n;++i)
        if (memcmp(haystack+i,needle,needle_n)==0) return (ptrdiff_t)i;
    return -1;
}
