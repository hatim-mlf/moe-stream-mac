#include <stdint.h>
#include <stdlib.h>
int parse_port_decimal(const char*t,uint16_t*out){if(!t||!out||!*t)return -1;char*end;unsigned long v=strtoul(t,&end,10);if(*end||v>65535)return -1;*out=(uint16_t)v;return 0;}
