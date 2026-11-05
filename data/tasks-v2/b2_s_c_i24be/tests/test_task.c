#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include "../candidate.c"
int main(void){uint8_t b[]={0x01,0x23,0x45,0xFF,0xFF,0xFE};int32_t v=7;
 assert(decode_i24_be(b,sizeof b,0,&v)==0 && v==0x12345);
 assert(decode_i24_be(b,sizeof b,3,&v)==0 && v==-2);
 assert(decode_i24_be(b,sizeof b,4,&v)==-1 && v==-2);
 assert(decode_i24_be(b,sizeof b,SIZE_MAX,&v)==-1 && v==-2);
 assert(decode_i24_be(NULL,0,0,&v)==-1 && v==-2);
 return 0;}
