#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include "../candidate.c"
int main(void){int8_t out=9;
 assert(requantize_i32_s8(1,1,1,0,&out)==0&&out==0);
 assert(requantize_i32_s8(3,1,1,0,&out)==0&&out==2);
 assert(requantize_i32_s8(-3,1,1,0,&out)==0&&out==-2);
 assert(requantize_i32_s8(100,4,0,-3,&out)==0&&out==127);
 assert(requantize_i32_s8(-100,4,0,3,&out)==0&&out==-128);
 out=7;assert(requantize_i32_s8(1,1,32,0,&out)==-1&&out==7);
 assert(requantize_i32_s8(1,-1,1,0,&out)==-1&&out==7);
 assert(requantize_i32_s8(1,1,1,128,&out)==-1&&out==7);
 return 0;
}
