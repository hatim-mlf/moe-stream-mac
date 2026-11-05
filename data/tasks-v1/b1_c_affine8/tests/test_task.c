#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    uint8_t q=0;
    assert(quantize_affine_u8(1.25f,0.5f,10,&q)==0 && q==13);
    assert(quantize_affine_u8(-2.0f,1.0f,10,&q)==0 && q==8);
    assert(quantize_affine_u8(1000.0f,1.0f,0,&q)==0 && q==255);
    q=44; assert(quantize_affine_u8(2,0,4,&q)==-1 && q==44);
    assert(quantize_affine_u8(2,1,300,&q)==-1 && q==44);
}
