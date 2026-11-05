#include <assert.h>
#include <stdint.h>
#include <math.h>
#include "../candidate.c"
int main(void) {
    int16_t q=7;
    assert(q8_8_encode(1.5f,&q)==0 && q==384);
    assert(q8_8_encode(0.5f/256.0f,&q)==0 && q==1);
    assert(q8_8_encode(-0.5f/256.0f,&q)==0 && q==-1);
    assert(q8_8_encode(127.99609375f,&q)==0 && q==32767);
    q=77; assert(q8_8_encode(128.0f,&q)==-1 && q==77);
    assert(q8_8_encode(NAN,&q)==-1 && q==77);
}
