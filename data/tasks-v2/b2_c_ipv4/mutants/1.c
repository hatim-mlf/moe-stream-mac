#include <stdint.h>
int parse_ipv4_u32(const char*t,uint32_t*out){if(!t||!out)return -1;uint32_t v=0;const unsigned char*p=(const unsigned char*)t;for(unsigned i=0;i<4;i++){if(*p<'0'||*p>'9')return -1;unsigned n=0;while(*p>='0'&&*p<='9'){n=n*10u+(unsigned)(*p++-'0');if(n>255)return -1;}v=(v<<8)|n;if(i<3){if(*p++!='.')return -1;}else if(*p)return -1;}*out=v;return 0;}
