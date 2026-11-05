#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){int32_t x[]={1,1,2,2,-1};assert(count_transitions_i32(x,5)==2);assert(count_transitions_i32(NULL,1)==0);assert(count_transitions_i32(NULL,0)==0);return 0;}
