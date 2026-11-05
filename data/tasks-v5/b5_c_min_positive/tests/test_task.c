#include <assert.h>
#include "../candidate.c"
int main(void){float x[]={0,-2,4,0.5f,1},y=9;assert(min_positive_f32(x,5,&y)==0&&y==0.5f);y=8;assert(min_positive_f32(NULL,0,&y)==-1&&y==8);float z[]={-1,0};assert(min_positive_f32(z,2,&y)==-1&&y==8);return 0;}
