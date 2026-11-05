#include <limits.h>
#include <stddef.h>
#include <stdint.h>
int hex_encode_lower(const uint8_t *s,size_t n,char *d,size_t c){static const char h[]="0123456789ABCDEF";if(!d||(n&&!s)||c<2*n+1||2*n>INT_MAX)return -1;for(size_t i=0;i<n;i++){d[2*i]=h[s[i]>>4];d[2*i+1]=h[s[i]&15];}d[2*n]=0;return (int)(2*n);}
