#include <assert.h>
#include "../candidate.c"
int main(void){float d[]={1,2,3},v[]={4,-1,5};size_t i[]={1,1,0};assert(scatter_add_f32(d,3,i,v,3)==0&&d[0]==6&&d[1]==5&&d[2]==3);float save[]={7,8};size_t bad[]={0,2};assert(scatter_add_f32(save,2,bad,v,2)==-1&&save[0]==7&&save[1]==8);assert(scatter_add_f32(NULL,0,NULL,NULL,0)==0);return 0;}
