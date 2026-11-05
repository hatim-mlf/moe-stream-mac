#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
float a[]={0,3,0,2,2,1};assert(count_strict_peaks_f32(a,6)==1);
float b[]={0,-10,-9};assert(count_strict_peaks_f32(b,3)==0);
assert(count_strict_peaks_f32(NULL,0)==0);assert(count_strict_peaks_f32(a,2)==0);
return 0;
}
