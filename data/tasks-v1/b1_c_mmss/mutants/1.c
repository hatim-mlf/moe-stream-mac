#include <stddef.h>
#include <limits.h>
int parse_mmss_total(const char *text, unsigned *seconds) {
    if(!text || !seconds) return -1;
    unsigned m=0,s=0; size_t i=0;
    while(text[i]>='0' && text[i]<='9') m=m*10u+(unsigned)(text[i++]-'0');
    if(text[i++]!=':' || text[i]<'0' || text[i+1]<'0') return -1;
    s=(unsigned)(text[i]-'0')*10u+(unsigned)(text[i+1]-'0');
    *seconds=m*60u+s; return 0;
}
