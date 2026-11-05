#include <math.h>
float sigmoid_f32(float x){double z=exp((double)x);return (float)(z/(1.0+z));}
