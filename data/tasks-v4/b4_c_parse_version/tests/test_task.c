#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    uint16_t v[3] = {8,8,8};
    assert(parse_version_triplet("v0.12.65535", v) == 0 && v[0] == 0 && v[1] == 12 && v[2] == 65535);
    v[0]=8; v[1]=8; v[2]=8; assert(parse_version_triplet("v01.2.3", v) == -1);
    assert(v[0] == 8 && v[1] == 8 && v[2] == 8);
    assert(parse_version_triplet("v1.2.65536", v) == -1);
    assert(parse_version_triplet("v1.2.3x", v) == -1);
    return 0;
}
