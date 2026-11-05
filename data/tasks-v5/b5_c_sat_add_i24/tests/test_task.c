#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){assert(sat_add_i24(8388600,20)==8388607);assert(sat_add_i24(-8388600,-20)==-8388608);assert(sat_add_i24(4,-2)==2);return 0;}
