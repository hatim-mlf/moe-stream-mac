#include <math.h>
float sigmoid_f32(float x){if(x>=0.0f){double z=exp(-(double)x);return (float)(1.0/(1.0+z));}double z=exp((double)x);return (float)(z/(1.0+z));}
