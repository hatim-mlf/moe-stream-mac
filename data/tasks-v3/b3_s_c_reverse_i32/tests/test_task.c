#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
int32_t a[]={1,-2,3,4}; reverse_i32(a,4);
assert(a[0]==4&&a[1]==3&&a[2]==-2&&a[3]==1);
int32_t b[]={7};reverse_i32(b,1);assert(b[0]==7); reverse_i32(NULL,0);
return 0;
}
