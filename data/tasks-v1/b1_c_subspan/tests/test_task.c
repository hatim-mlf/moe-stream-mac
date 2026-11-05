#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    uint8_t h[]={1,2,3,2,3}, n[]={2,3}, bad[]={8};
    assert(find_subspan_u8(h,5,n,2)==1);
    assert(find_subspan_u8(h,5,bad,1)==-1);
    uint8_t tail[]={8,9,1}, end_needle[]={9,1};
    assert(find_subspan_u8(tail,3,end_needle,2)==1);
    assert(find_subspan_u8(h,0,NULL,0)==0);
    assert(find_subspan_u8(h,5,h,6)==-1);
}
