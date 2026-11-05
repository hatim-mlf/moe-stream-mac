#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
float a[]={0,3,-2,-1,4};assert(max_adjacent_gap_f32(a,5)==5.0f);
float b[]={0,-10,-9};assert(max_adjacent_gap_f32(b,3)==10.0f);
assert(max_adjacent_gap_f32(NULL,0)==0.0f);float c[]={4};assert(max_adjacent_gap_f32(c,1)==0.0f);
return 0;
}
