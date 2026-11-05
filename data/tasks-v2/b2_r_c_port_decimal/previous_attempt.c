#include <stdint.h>
#include <stdlib.h>
int parse_port_decimal(const char *text,uint16_t *out){
 if(!text||!out||!*text)return -1;
 char *end=NULL;unsigned long value=strtoul(text,&end,10);
 if(*end!='\0'||value>65535ul)return -1;
 *out=(uint16_t)value;return 0;
}