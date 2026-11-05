#include <math.h>
float gelu_tanh_approx_f32(float x){double z=x;double u=sqrt(2.0/3.14159265358979323846)*(z+0.1*z*z*z);return (float)(0.5*z*(1.0+tanh(u)));}
