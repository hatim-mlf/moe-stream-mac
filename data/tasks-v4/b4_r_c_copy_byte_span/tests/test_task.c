#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    uint8_t b[6] = {1,2,3,4,5,6};
    assert(copy_byte_span(b, 6, 0, b+1, 5, 4) == 0);
    assert(b[0]==1 && b[1]==1 && b[2]==2 && b[3]==3 && b[4]==4);
    uint8_t c[4]={9,8,7,6};
    assert(copy_byte_span(c,4,3,c,4,2)==-1 && c[0]==9 && c[3]==6);
    assert(copy_byte_span(NULL,0,0,NULL,0,0)==0);
    return 0;
}
