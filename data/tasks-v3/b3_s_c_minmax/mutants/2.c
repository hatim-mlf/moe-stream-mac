#include <stdint.h>
#include <stddef.h>
int minmax_i16(const int16_t*x,size_t n,int16_t*lo,int16_t*hi){if(n==0||!x||!lo||!hi)return -1;int16_t a=x[0],b=x[0];for(size_t i=1;i<n;i++){if(x[i]<a)a=x[i];if(x[i]>b)b=x[i];}*lo=b;*hi=a;return 0;}
