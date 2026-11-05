#include <assert.h>
#include <stddef.h>
#include "../candidate.c"
int main(void){float x[]={100,9,9,-1,8};
 assert(first_max_window_f32(x,5,1,3)==1);
 assert(first_max_window_f32(x,5,2,3)==2);
 assert(first_max_window_f32(x,5,5,0)==-1);
 assert(first_max_window_f32(x,5,4,2)==-1);
 assert(first_max_window_f32(NULL,0,0,1)==-1);
 return 0;}
