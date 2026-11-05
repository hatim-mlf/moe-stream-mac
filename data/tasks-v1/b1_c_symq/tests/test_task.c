#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    int8_t q=0;
    assert(quantize_symmetric_s8(1.5f,1,&q)==0 && q==2);
    assert(quantize_symmetric_s8(-1.5f,1,&q)==0 && q==-2);
    assert(quantize_symmetric_s8(-900,1,&q)==0 && q==-127);
    q=12; assert(quantize_symmetric_s8(1,0,&q)==-1 && q==12);
}
