#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include "../candidate.c"
int main(void){int32_t out=17;
 assert(q16_16_divide_sat(65536,131072,&out)==0&&out==32768);
 assert(q16_16_divide_sat(1,131072,&out)==0&&out==0);
 assert(q16_16_divide_sat(3,131072,&out)==0&&out==2);
 assert(q16_16_divide_sat(-3,131072,&out)==0&&out==-2);
 assert(q16_16_divide_sat(INT32_MAX,1,&out)==0&&out==INT32_MAX);
 assert(q16_16_divide_sat(INT32_MIN,1,&out)==0&&out==INT32_MIN);
 out=9;assert(q16_16_divide_sat(1,0,&out)==-1&&out==9);
 assert(q16_16_divide_sat(1,1,NULL)==-1);
 return 0;
}
