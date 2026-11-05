#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    uint8_t s[]={4,4,9,9,9,2},v[3]={0}; uint32_t c[3]={0}; size_t n=88;
    assert(encode_runs_u8(s,6,v,c,3,&n)==0 && n==3);
    assert(v[0]==4 && c[0]==2 && v[1]==9 && c[1]==3 && v[2]==2 && c[2]==1);
    n=77; v[0]=55;
    assert(encode_runs_u8(s,6,v,c,2,&n)==-1 && n==77 && v[0]==55);
    assert(encode_runs_u8(NULL,0,NULL,NULL,0,&n)==0 && n==0);
}
