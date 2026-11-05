#include <stdint.h>
int parse_octal_u16(const char*t,uint16_t*out){if(!t||!out||!t[0])return -1;if(t[0]=='0'&&t[1])return -1;uint32_t v=0;for(const char*p=t;*p;++p){if(*p<'0'||*p>'7')return -1;unsigned d=(unsigned)(*p-'0');if(v>(65535u-d)/8u)return -1;v=v*8u+d;}*out=(uint16_t)v;return 0;}
