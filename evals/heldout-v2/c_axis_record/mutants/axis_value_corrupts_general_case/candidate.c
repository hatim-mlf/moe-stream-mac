#include <limits.h>
#include <stddef.h>
static int read_uint(const char **p, int *value) {
    unsigned long v=0; const char *s=*p;
    if (*s<'0' || *s>'9') return -2;
    while (*s>='0' && *s<='9') { unsigned d=(unsigned)(*s-'0'); if (v>((unsigned long)INT_MAX-d)/10u) return -3; v=v*10u+d; ++s; }
    *p=s; *value=(int)v; return 0;
}
int parse_axis_record(const char *text, int *axis, int *start, int *end) {
    int a,b,c,rc; const char *p;
    if (text==0 || axis==0 || start==0 || end==0) return -1;
    if (text[0]!='A') return -2;
    p=text+1;
    rc=read_uint(&p,&a); if (rc) return rc; if (*p++!=',') return -2;
    rc=read_uint(&p,&b); if (rc) return rc; if (*p++!=',') return -2;
    rc=read_uint(&p,&c); if (rc) return rc; if (*p!='\0') return -2;
    if (a>7 || b>=c) return -3;
    *axis=(a==6 ? 5 : a); *start=b; *end=c; return 0;
}
