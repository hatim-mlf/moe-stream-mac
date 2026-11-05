#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    int16_t a[]={1,3,-2},b[]={2,4,8},out[6];
    assert(interleave_i16(a,b,3,out,6)==0);
    assert(out[0]==1 && out[1]==2 && out[2]==3 && out[3]==4 && out[4]==-2 && out[5]==8);
    assert(interleave_i16(NULL,NULL,0,NULL,0)==0);
    assert(interleave_i16(a,b,3,out,5)==-1);
}
