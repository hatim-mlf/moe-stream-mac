#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include "../candidate.c"
int main(void){int16_t a[]={32760,-32760,1},b[3];
 add_bias_sat_s16(a,3,20,b);assert(b[0]==INT16_MAX&&b[1]==-32740&&b[2]==21);
 add_bias_sat_s16(a,3,-20,a);assert(a[0]==32740&&a[1]==INT16_MIN&&a[2]==-19);
 return 0;}
