#include <stdint.h>
#include <stddef.h>
void prefix_min_i16(const int16_t*s,int16_t*d,size_t n){if(!n)return;d[0]=0;int16_t b=s[0];for(size_t i=1;i<n;i++){if(s[i]<b)b=s[i];d[i]=b;}}
