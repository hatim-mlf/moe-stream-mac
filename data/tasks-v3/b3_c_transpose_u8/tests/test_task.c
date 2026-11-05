#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
uint8_t a[]={1,2,3,4,5,6},b[6]={0}; transpose_u8(a,2,3,b);
assert(b[0]==1&&b[1]==4&&b[2]==2&&b[3]==5&&b[4]==3&&b[5]==6);
transpose_u8(NULL,0,4,NULL); transpose_u8(NULL,4,0,NULL);
return 0;
}
