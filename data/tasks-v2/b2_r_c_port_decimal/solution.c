#include <stdint.h>
int parse_port_decimal(const char *text,uint16_t *out){
 if(!text||!out||*text<'0'||*text>'9')return -1;
 unsigned value=0;for(const unsigned char*p=(const unsigned char*)text;*p;p++){
  if(*p<'0'||*p>'9')return -1;
  value=value*10u+(unsigned)(*p-'0');if(value>65535u)return -1;
 }
 *out=(uint16_t)value;return 0;
}
