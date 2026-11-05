#include <assert.h>
#include <stddef.h>
#include "../candidate.c"
int main(void) {
    const unsigned char a[] = {'A',' ','~'};
    const unsigned char b[] = {'a',0x1f};
    const unsigned char c[] = {0x7f};
    assert(is_printable_ascii_span(a,3)==1);
    assert(is_printable_ascii_span(b,2)==0 && is_printable_ascii_span(c,1)==0);
    assert(is_printable_ascii_span(NULL,0)==1 && is_printable_ascii_span(NULL,1)==0);
    return 0;
}
