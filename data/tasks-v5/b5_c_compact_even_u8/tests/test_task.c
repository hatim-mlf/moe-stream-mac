#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){uint8_t x[]={4,3,2,2,7,8};size_t n=compact_even_u8(x,6);assert(n==4&&x[0]==4&&x[1]==2&&x[2]==2&&x[3]==8);assert(compact_even_u8(NULL,0)==0);return 0;}
