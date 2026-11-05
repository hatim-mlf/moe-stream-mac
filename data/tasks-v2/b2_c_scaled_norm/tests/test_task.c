#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void){float a[]={3,4},b[]={1.0e30f,1.0e30f},z[]={0,0};
 assert(fabs(l2_norm_f32(a,2)-5.0)<1e-12);
 assert(fabs(l2_norm_f32(b,2)/1.0e30-sqrt(2.0))<1e-6);
 assert(l2_norm_f32(z,2)==0 && l2_norm_f32(NULL,0)==0);
 return 0;}
