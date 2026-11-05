#include <stdint.h>
int parse_ipv4_u32(const char *text,uint32_t *out) {
    if(!text||!out)return -1;
    uint32_t value=0;
    const unsigned char *p=(const unsigned char*)text;
    for(unsigned part=0;part<4;part++) {
        if(*p<'0'||*p>'9')return -1;
        if(*p=='0' && p[1]>='0' && p[1]<='9')return -1;
        unsigned v=0,digits=0;
        while(*p>='0'&&*p<='9') { if(++digits>3)return -1; v=v*10u+(unsigned)(*p++-'0'); }
        if(v>255)return -1;
        value=(value<<8)|v;
        if(part<3) { if(*p++!='.')return -1; }
        else if(*p!='\0')return -1;
    }
    *out=value;return 0;
}
