#include <assert.h>
#include <stddef.h>
#include "../candidate.c"
int main(void){assert(reflect_index(-1,4)==0);assert(reflect_index(-2,4)==1);assert(reflect_index(3,4)==3);assert(reflect_index(4,4)==3);assert(reflect_index(5,4)==2);assert(reflect_index(-99,1)==0);return 0;}
