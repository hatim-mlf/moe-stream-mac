#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include "../candidate.c"
int main(void){uint8_t d[]={0x34,0x12,0xaa};uint16_t x=9;assert(read_u16le_at(d,3,0,&x)==0&&x==0x1234);assert(read_u16le_at(d,3,1,&x)==0&&x==0xaa12);x=7;assert(read_u16le_at(d,3,2,&x)==-1&&x==7);return 0;}
