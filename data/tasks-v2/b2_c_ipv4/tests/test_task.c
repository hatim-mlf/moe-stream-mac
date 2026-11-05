#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){uint32_t v=9;
 assert(parse_ipv4_u32("192.168.1.10",&v)==0 && v==UINT32_C(0xC0A8010A));
 assert(parse_ipv4_u32("0.0.0.0",&v)==0 && v==0);
 assert(parse_ipv4_u32("255.255.255.255",&v)==0 && v==UINT32_MAX);
 assert(parse_ipv4_u32("01.2.3.4",&v)==-1 && v==UINT32_MAX);
 assert(parse_ipv4_u32("1.2.3.256",&v)==-1 && v==UINT32_MAX);
 assert(parse_ipv4_u32("1.2.3",&v)==-1 && v==UINT32_MAX);
 assert(parse_ipv4_u32("1.2.3.4x",&v)==-1 && v==UINT32_MAX);
 return 0;}
