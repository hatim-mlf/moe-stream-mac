#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
float a[]={1,2,3,4,5,6};assert(trace_rowmajor_f32(a,2,3)==6.0f);
float b[]={2,9,8,4};assert(trace_rowmajor_f32(b,2,2)==6.0f);
assert(trace_rowmajor_f32(NULL,0,4)==0.0f);assert(trace_rowmajor_f32(NULL,3,0)==0.0f);
return 0;
}
