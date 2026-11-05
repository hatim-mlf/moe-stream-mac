#include <assert.h>
#include "../candidate.c"
int main(void) {
    for (unsigned char c=9;c<=13;c++) assert(is_ascii_space_byte(c)==1);
    assert(is_ascii_space_byte(32)==1 && is_ascii_space_byte(0)==0 && is_ascii_space_byte(160)==0);
    return 0;
}
