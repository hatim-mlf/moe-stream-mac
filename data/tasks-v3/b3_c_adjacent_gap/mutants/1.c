#include <stddef.h>
#include <math.h>
float max_adjacent_gap_f32(const float*x,size_t n){if(n<2)return 0;float best=x[1]-x[0];for(size_t i=2;i<n;i++)if(x[i]-x[i-1]>best)best=x[i]-x[i-1];return best;}
