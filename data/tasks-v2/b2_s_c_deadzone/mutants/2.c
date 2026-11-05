#include <math.h>
float deadzone_f32(float x,float t){if(!isfinite(x)||!isfinite(t)||t<0)return 0;if(fabsf(x)<=t)return x;return x>0?x-t:x+t;}
