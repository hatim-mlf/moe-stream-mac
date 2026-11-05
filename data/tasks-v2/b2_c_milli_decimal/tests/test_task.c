#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include "../candidate.c"
int main(void){int64_t v=7;
 assert(parse_milli_i64("+12.34",&v)==0 && v==12340);
 assert(parse_milli_i64("-0.007",&v)==0 && v==-7);
 assert(parse_milli_i64("9223372036854775.807",&v)==0 && v==INT64_MAX);
 assert(parse_milli_i64("-9223372036854775.808",&v)==0 && v==INT64_MIN);
 assert(parse_milli_i64(".5",&v)==-1 && v==INT64_MIN);
 assert(parse_milli_i64("2.",&v)==-1 && v==INT64_MIN);
 assert(parse_milli_i64("1.0000",&v)==-1 && v==INT64_MIN);
 assert(parse_milli_i64("9223372036854775.808",&v)==-1 && v==INT64_MIN);
 return 0;}
