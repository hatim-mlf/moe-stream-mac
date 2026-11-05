#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    uint8_t a[]={1,2,3,4,5};
    assert(rotate_buffer_left(a,5,2)==0);
    assert(a[0]==3 && a[1]==4 && a[2]==5 && a[3]==1 && a[4]==2);
    uint8_t b[]={7,8};
    assert(rotate_buffer_left(b,2,8)==0 && b[0]==7 && b[1]==8);
    assert(rotate_buffer_left(NULL,0,99)==0);
    assert(rotate_buffer_left(NULL,1,0)==-1);
}
