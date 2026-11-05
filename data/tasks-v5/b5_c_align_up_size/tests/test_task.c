#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include "../candidate.c"
int main(void){size_t x=99;assert(align_up_size(10,6,&x)==0&&x==12);assert(align_up_size(12,6,&x)==0&&x==12);x=77;assert(align_up_size(SIZE_MAX,2,&x)==-1&&x==77);assert(align_up_size(3,0,&x)==-1);return 0;}
