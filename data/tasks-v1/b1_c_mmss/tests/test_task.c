#include <assert.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
    unsigned s=9;
    assert(parse_mmss_total("0:00",&s)==0 && s==0);
    assert(parse_mmss_total("12:34",&s)==0 && s==754);
    assert(parse_mmss_total("0002:05",&s)==0 && s==125);
    s=91; assert(parse_mmss_total("3:60",&s)==-1 && s==91);
    assert(parse_mmss_total("3:7",&s)==-1 && s==91);
    assert(parse_mmss_total(" 3:07",&s)==-1 && s==91);
    assert(parse_mmss_total("4294967295:00",&s)==-1 && s==91);
}
