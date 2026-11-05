#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include "../candidate.c"
int main(void){
 assert(midpoint_floor_i32(3,4)==3); assert(midpoint_floor_i32(-3,0)==-2);
 assert(midpoint_floor_i32(INT32_MIN,INT32_MAX)==-1);
 assert(midpoint_floor_i32(INT32_MIN,INT32_MIN)==INT32_MIN);
 assert(midpoint_floor_i32(INT32_MAX,INT32_MAX)==INT32_MAX);
 return 0;}
