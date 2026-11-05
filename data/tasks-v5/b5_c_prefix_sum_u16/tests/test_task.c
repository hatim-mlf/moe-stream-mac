#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){uint16_t x[]={65530,10,2};prefix_sum_u16_mod(x,x,3);assert(x[0]==65530&&x[1]==4&&x[2]==6);prefix_sum_u16_mod(NULL,NULL,0);return 0;}
