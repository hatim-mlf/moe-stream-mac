#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){uint8_t q=99;assert(quantize_sqrt_u8(0,&q)==0&&q==0);assert(quantize_sqrt_u8(.25f,&q)==0&&q==128);assert(quantize_sqrt_u8(1,&q)==0&&q==255);q=5;assert(quantize_sqrt_u8(-.1f,&q)==-1&&q==5);assert(quantize_sqrt_u8(1.1f,&q)==-1&&q==5);return 0;}
