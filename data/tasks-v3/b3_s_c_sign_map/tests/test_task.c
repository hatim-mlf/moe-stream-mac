#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
int16_t a[]={-9,0,4,INT16_MIN};int8_t b[4];sign_map_i16(a,b,4);
assert(b[0]==-1&&b[1]==0&&b[2]==1&&b[3]==-1);sign_map_i16(NULL,NULL,0);
return 0;
}
