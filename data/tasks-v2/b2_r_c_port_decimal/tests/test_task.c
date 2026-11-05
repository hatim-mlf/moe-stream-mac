#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){uint16_t v=7;
 assert(parse_port_decimal("0",&v)==0 && v==0);
 assert(parse_port_decimal("65535",&v)==0 && v==65535);
 assert(parse_port_decimal("+80",&v)==-1 && v==65535);
 assert(parse_port_decimal(" 80",&v)==-1 && v==65535);
 assert(parse_port_decimal("80 ",&v)==-1 && v==65535);
 assert(parse_port_decimal("65536",&v)==-1 && v==65535);
 return 0;}
