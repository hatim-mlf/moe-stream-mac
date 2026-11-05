#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){int8_t x[]={-128,-3,0,4};assert(sum_squares_i8(x,4)==16409);assert(sum_squares_i8(NULL,0)==0);return 0;}
