#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){int32_t x[]={2,3,4,-1};int64_t out=99;
 assert(sum_i32_above(x,4,3,&out)==0 && out==4);
 assert(sum_i32_above(x,4,2,&out)==0 && out==7);
 assert(sum_i32_above(NULL,0,0,&out)==0 && out==0);
 out=12;assert(sum_i32_above(NULL,1,0,&out)==-1 && out==12);
 return 0;}
