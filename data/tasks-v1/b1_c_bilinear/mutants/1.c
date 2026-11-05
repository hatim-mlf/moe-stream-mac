#include <stddef.h>
#include <math.h>
float bilinear_sample_f32(const float *g,size_t r,size_t c,float y,float x) {
    if(y<0)y=0; if(x<0)x=0; if(y>(float)(r-1))y=(float)(r-1); if(x>(float)(c-1))x=(float)(c-1);
    size_t y0=(size_t)floorf(y),x0=(size_t)floorf(x),y1=y0+1<r?y0+1:y0,x1=x0+1<c?x0+1:x0;
    return g[y0*c+x0]*(1-y)*(1-x)+g[y0*c+x1]*(1-y)*x+g[y1*c+x0]*y*(1-x)+g[y1*c+x1]*y*x;
}
