#include <stddef.h>
#include <math.h>
float bilinear_sample_f32(const float *grid,size_t rows,size_t cols,float y,float x) {
    if (y<0.0f) y=0.0f;
    if (x<0.0f) x=0.0f;
    if (y>(float)(rows-1)) y=(float)(rows-1);
    if (x>(float)(cols-1)) x=(float)(cols-1);
    size_t y0=(size_t)floorf(y), x0=(size_t)floorf(x);
    size_t y1=y0+1<rows ? y0+1 : y0;
    size_t x1=x0+1<cols ? x0+1 : x0;
    float fy=y-(float)y0, fx=x-(float)x0;
    float a=grid[y0*cols+x0]*(1.0f-fx)+grid[y0*cols+x1]*fx;
    float b=grid[y1*cols+x0]*(1.0f-fx)+grid[y1*cols+x1]*fx;
    return a*(1.0f-fy)+b*fy;
}
