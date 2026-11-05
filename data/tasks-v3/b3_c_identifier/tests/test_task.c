#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
assert(is_ascii_identifier("_a9")==1); assert(is_ascii_identifier("Q")==1);
assert(is_ascii_identifier("")==0&&is_ascii_identifier("9a")==0);
assert(is_ascii_identifier("a-b")==0&&is_ascii_identifier(NULL)==0);
char hi[]={ (char)0xE9,0 }; assert(is_ascii_identifier(hi)==0);
return 0;
}
