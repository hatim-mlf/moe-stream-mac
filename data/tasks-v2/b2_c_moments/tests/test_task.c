#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void){float x[]={1,2,3};double m=-1,v=-1;
 assert(mean_popvar_f32(x,3,&m,&v)==0 && fabs(m-2)<1e-12 && fabs(v-2.0/3.0)<1e-12);
 x[0]=4;assert(mean_popvar_f32(x,1,&m,&v)==0 && m==4 && v==0);
 m=9;v=8;assert(mean_popvar_f32(x,0,&m,&v)==-1 && m==9 && v==8);
 assert(mean_popvar_f32(NULL,1,&m,&v)==-1 && m==9 && v==8);
 return 0;}
