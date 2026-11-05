#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){uint16_t x=99;assert(parse_octal_u16("177777",&x)==0&&x==65535);assert(parse_octal_u16("0",&x)==0&&x==0);x=9;assert(parse_octal_u16("128",&x)==-1&&x==9);assert(parse_octal_u16("010",&x)==-1&&x==9);assert(parse_octal_u16("200000",&x)==-1&&x==9);return 0;}
