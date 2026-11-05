#include <stddef.h>
#include <math.h>
float max_adjacent_gap_f32(const float*x,size_t n){if(n<2)return 0;float lo=x[0],hi=x[0];for(size_t i=1;i<n;i++){if(x[i]<lo)lo=x[i];if(x[i]>hi)hi=x[i];}return hi-lo;}
