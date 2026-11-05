#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include "../candidate.c"
int main(void) {
 uint32_t out=99;
 assert(extract_u32_field(UINT32_C(0xD6A5),4,8,&out)==0 && out==0x6A);
 assert(extract_u32_field(UINT32_C(0x80000001),31,1,&out)==0 && out==1);
 assert(extract_u32_field(UINT32_C(0x89ABCDEF),0,32,&out)==0 && out==UINT32_C(0x89ABCDEF));
 assert(extract_u32_field(0,0,0,&out)==-1 && out==UINT32_C(0x89ABCDEF));
 assert(extract_u32_field(0,31,2,&out)==-1 && out==UINT32_C(0x89ABCDEF));
 assert(extract_u32_field(0,0,1,NULL)==-1);
 return 0;
}
