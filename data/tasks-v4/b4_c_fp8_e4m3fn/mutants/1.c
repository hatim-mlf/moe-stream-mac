#include <math.h>
#include <stdint.h>
uint8_t float_to_e4m3fn(float x){uint8_t s=x<0?128:0;double v=fabs(x);unsigned e=0;while(e<126){unsigned ex=e>>3,m=e&7;double y=ex==0?m/512.0:ldexp(1.0+m/8.0,(int)ex-7);if(y>=v)break;e++;}return (uint8_t)(s|e);}
