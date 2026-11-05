#include <stddef.h>
#include <stdint.h>
#include <limits.h>
ptrdiff_t first_max_window_f32(const float*x,size_t n,size_t start,size_t width){if(!x||width==0||start>n||width>n-start)return -1;size_t best=0;for(size_t i=start;i<start+width;i++)if(x[i]>x[best])best=i;if(best>(size_t)PTRDIFF_MAX)return -1;return (ptrdiff_t)best;}
