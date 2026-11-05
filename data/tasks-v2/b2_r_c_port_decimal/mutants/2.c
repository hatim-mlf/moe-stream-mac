#include <stdint.h>
int parse_port_decimal(const char*t,uint16_t*out){if(!t||!out||*t<'0'||*t>'9')return -1;unsigned v=0;for(;*t;t++){if(*t<'0'||*t>'9')return -1;v=v*10u+(unsigned)(*t-'0');if(v>65534)return -1;}*out=(uint16_t)v;return 0;}
