#include <stdint.h>
#include <stddef.h>
#include <limits.h>
int parse_i16_csv(const char *text,int16_t *out,size_t cap,size_t *count) {
    if(text==NULL || count==NULL) return -1;
    size_t n=0; const char *p=text;
    while(*p) {
        int neg=0; unsigned v=0,limit;
        if(*p=='+' || *p=='-') { neg=*p=='-'; ++p; }
        if(*p<'0'||*p>'9') return -1;
        limit=neg?32768u:32767u;
        if(*p=='0' && p[1]>='0' && p[1]<='9') return -1;
        do { unsigned d=(unsigned)(*p-'0'); if(v>(limit-d)/10u) return -1; v=v*10u+d; ++p; } while(*p>='0'&&*p<='9');
        if(*p!=',' && *p!=0) return -1;
        ++n;
        if(*p==',') { ++p; if(*p==0) return -1; }
    }
    if(n==0 || n>cap || (n && out==NULL) || n>(size_t)INT_MAX) return -1;
    p=text;
    for(size_t i=0;i<n;i++) {
        int neg=0; int v=0;
        if(*p=='+'||*p=='-'){neg=*p=='-';++p;}
        while(*p>='0'&&*p<='9'){v=v*10+(*p-'0');++p;}
        out[i]=(int16_t)(neg?-v:v);
        if(*p==',') ++p;
    }
    *count=n;
    return (int)n;
}
