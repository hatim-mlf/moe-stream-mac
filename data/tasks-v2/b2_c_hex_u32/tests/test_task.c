#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){uint32_t v=17;
 assert(parse_hex_u32_exact("fF01",&v)==0 && v==0xFF01u);
 assert(parse_hex_u32_exact("FFFFFFFF",&v)==0 && v==UINT32_MAX);
 assert(parse_hex_u32_exact("",&v)==-1 && v==UINT32_MAX);
 assert(parse_hex_u32_exact("0x12",&v)==-1 && v==UINT32_MAX);
 assert(parse_hex_u32_exact("123456789",&v)==-1 && v==UINT32_MAX);
 assert(parse_hex_u32_exact("12 ",&v)==-1 && v==UINT32_MAX);
 return 0;}
