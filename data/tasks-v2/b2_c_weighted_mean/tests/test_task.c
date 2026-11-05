#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void){float x[]={2,8,10},w[]={1,3,0};double out=-9;
 assert(weighted_mean_f32(x,w,3,&out)==0 && fabs(out-6.5)<1e-12);
 w[0]=w[1]=w[2]=0;out=17;assert(weighted_mean_f32(x,w,3,&out)==-1 && out==17);
 w[0]=1; x[1]=NAN;assert(weighted_mean_f32(x,w,3,&out)==-1 && out==17);
 assert(weighted_mean_f32(x,w,0,&out)==-1 && out==17);
 return 0;}
