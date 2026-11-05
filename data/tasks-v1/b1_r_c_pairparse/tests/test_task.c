#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    int32_t a=55,b=66;
    assert(parse_i32_pair("-12,+7",&a,&b)==0 && a==-12 && b==7);
    a=55;b=66; assert(parse_i32_pair("4,5tail",&a,&b)==-1 && a==55 && b==66);
    assert(parse_i32_pair(" 1,2",&a,&b)==-1);
    assert(parse_i32_pair("1,2 ",&a,&b)==-1);
    assert(parse_i32_pair("2147483648,0",&a,&b)==-1);
}
