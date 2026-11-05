#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){uint8_t x[]={2,5,9};uint64_t s=88;assert(sum_u8_range(x,3,1,3,&s)==0&&s==14);s=88;assert(sum_u8_range(x,3,2,1,&s)==-1&&s==88);assert(sum_u8_range(NULL,0,0,0,&s)==0&&s==0);return 0;}
