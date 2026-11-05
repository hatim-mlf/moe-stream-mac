#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
int v=8; assert(parse_ascii_bool("true",&v)==0&&v==1);
assert(parse_ascii_bool("0",&v)==0&&v==0); v=8;
assert(parse_ascii_bool("True",&v)==-1&&v==8);
assert(parse_ascii_bool("true ",&v)==-1&&v==8);
assert(parse_ascii_bool(NULL,&v)==-1&&v==8);
return 0;
}
