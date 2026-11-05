#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){assert(clamp_i16_value(-9,-4,5)==-4);assert(clamp_i16_value(7,-4,5)==5);assert(clamp_i16_value(2,-4,5)==2);return 0;}
