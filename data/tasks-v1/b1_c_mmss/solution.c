#include <stddef.h>
#include <limits.h>
int parse_mmss_total(const char *text, unsigned *seconds) {
    if (!text || !seconds || text[0] < '0' || text[0] > '9') return -1;
    unsigned minutes=0;
    size_t i=0;
    while (text[i]>='0' && text[i]<='9') {
        unsigned digit=(unsigned)(text[i]-'0');
        if (minutes > (UINT_MAX-digit)/10u) return -1;
        minutes=minutes*10u+digit;
        ++i;
    }
    if (text[i++]!=':' || text[i]<'0' || text[i]>'9' ||
        text[i+1]<'0' || text[i+1]>'9' || text[i+2]!='\0') return -1;
    unsigned sec=(unsigned)(text[i]-'0')*10u+(unsigned)(text[i+1]-'0');
    if (sec>59u || minutes>(UINT_MAX-sec)/60u) return -1;
    *seconds=minutes*60u+sec;
    return 0;
}
