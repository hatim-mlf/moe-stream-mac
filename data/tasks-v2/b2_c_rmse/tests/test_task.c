#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void){float a[]={1,4,-2},b[]={1,1,2};double out=-1;
 assert(rmse_f32(a,b,3,&out)==0 && fabs(out-sqrt(25.0/3.0))<1e-12);
 assert(rmse_f32(a,b,1,&out)==0 && out==0.0);
 out=19; assert(rmse_f32(a,b,0,&out)==-1 && out==19);
 assert(rmse_f32(NULL,b,1,&out)==-1 && out==19);
 assert(rmse_f32(a,b,3,NULL)==-1);
 return 0;}
