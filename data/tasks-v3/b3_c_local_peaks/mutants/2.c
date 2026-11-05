#include <stddef.h>
size_t count_strict_peaks_f32(const float*x,size_t n){size_t c=0;for(size_t i=0;i<n;i++)if((i==0||x[i]>x[i-1])&&(i+1==n||x[i]>x[i+1]))c++;return c;}
