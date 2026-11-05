#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void){float x[]={1,2,3},h[]={2,1},y[3];fir_causal_f32(x,3,h,2,y);assert(y[0]==2&&y[1]==5&&y[2]==8);fir_causal_f32(x,3,NULL,0,y);assert(y[0]==0&&y[2]==0);fir_causal_f32(NULL,0,NULL,0,NULL);return 0;}
