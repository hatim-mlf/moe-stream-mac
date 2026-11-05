#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void){float a[]={0,0},b[]={1000,1000},c[]={-1000,0};
 assert(fabs(logsumexp_f32(a,2)-log(2.0))<1e-12);
 assert(fabs(logsumexp_f32(b,2)-(1000.0+log(2.0)))<1e-12);
 assert(fabs(logsumexp_f32(c,2))<1e-12);
 return 0;}
