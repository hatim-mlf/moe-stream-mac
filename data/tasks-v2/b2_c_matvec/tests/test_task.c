#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void){float a[]={1,2,3,4,5,6},x[]={2,-1,0.5f},y[2]={9,9};
 matvec_rowmajor_f32(a,x,2,3,y); assert(y[0]==1.5f && y[1]==6.0f);
 matvec_rowmajor_f32(NULL,NULL,2,0,y); assert(y[0]==0 && y[1]==0);
 matvec_rowmajor_f32(NULL,NULL,0,4,NULL);
 return 0;}
