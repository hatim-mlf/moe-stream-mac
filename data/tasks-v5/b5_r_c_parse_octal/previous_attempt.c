#include <stdint.h>
int parse_octal_u16(const char*t,uint16_t*out){if(!t||!out||!t[0])return -1;uint32_t v=0;for(const char*p=t;*p;++p){if(*p<'0'||*p>'9')return -1;v=v*8u+(unsigned)(*p-'0');if(v>65535)return -1;}*out=(uint16_t)v;return 0;}
