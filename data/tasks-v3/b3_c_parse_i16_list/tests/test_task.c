#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
int16_t o[4]={9,9,9,9}; size_t n=99;
assert(parse_i16_csv("-32768,+0,12,32767",o,4,&n)==4&&n==4);
assert(o[0]==INT16_MIN&&o[1]==0&&o[2]==12&&o[3]==INT16_MAX);
int16_t keep[2]={5,6}; n=44;
assert(parse_i16_csv("1,40000",keep,2,&n)==-1&&keep[0]==5&&keep[1]==6&&n==44);
assert(parse_i16_csv("01",keep,2,&n)==-1&&n==44);
assert(parse_i16_csv("1,2",keep,1,&n)==-1&&keep[0]==5&&n==44);
assert(parse_i16_csv("",keep,2,&n)==-1);
return 0;
}
