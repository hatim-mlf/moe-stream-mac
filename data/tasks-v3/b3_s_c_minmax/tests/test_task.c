#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
int16_t x[]={7,-4,9,-4},lo=1,hi=2;
assert(minmax_i16(x,4,&lo,&hi)==0&&lo==-4&&hi==9);
lo=1;hi=2;assert(minmax_i16(x,0,&lo,&hi)==-1&&lo==1&&hi==2);
assert(minmax_i16(x,4,NULL,&hi)==-1);
return 0;
}
