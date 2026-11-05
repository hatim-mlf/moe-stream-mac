#include <stdint.h>
#include <stddef.h>
int parse_route_slot(const char *text,char *name,size_t cap,uint16_t *slot) {
    if(!text||!name||!slot)return -1;
    size_t n=0; while(text[n] && text[n]!='@') {
        unsigned char c=(unsigned char)text[n];
        if((n==0 && (c<'a'||c>'z')) || (n>0 && !((c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_'))) return -1;
        ++n;
    }
    if(n==0||text[n]!='@'||cap<=n)return -1;
    const char *p=text+n+1; if(!*p || (*p=='0'&&p[1]))return -1;
    unsigned v=0;
    for(;*p;++p) { if(*p<'0'||*p>'9')return -1; unsigned d=(unsigned)(*p-'0'); if(v>(65535u-d)/10u)return -1; v=v*10u+d; }
    for(size_t i=0;i<n;++i)name[i]=text[i]; name[n]='\0'; *slot=(uint16_t)v; return 0;
}
