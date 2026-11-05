#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
float x[]={2,-1,-1,3};assert(argmin_first_f32(x,4)==1);
float y[]={5};assert(argmin_first_f32(y,1)==0);assert(argmin_first_f32(NULL,0)==-1);
return 0;
}
