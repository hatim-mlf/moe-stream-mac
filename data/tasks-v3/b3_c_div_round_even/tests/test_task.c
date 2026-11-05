#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
int64_t o=77;
assert(div_round_even_i64(5,2,&o)==0&&o==2);
assert(div_round_even_i64(7,2,&o)==0&&o==4);
assert(div_round_even_i64(-5,2,&o)==0&&o==-2);
assert(div_round_even_i64(-7,2,&o)==0&&o==-4);
assert(div_round_even_i64(5,-2,&o)==0&&o==-2);
o=77; assert(div_round_even_i64(1,0,&o)==-1&&o==77);
assert(div_round_even_i64(INT64_MIN,-1,&o)==-1&&o==77);
assert(div_round_even_i64(INT64_MIN,1,&o)==0&&o==INT64_MIN);
return 0;
}
