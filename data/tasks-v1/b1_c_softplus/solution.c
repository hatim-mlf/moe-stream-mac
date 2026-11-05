#include <math.h>
float softplus_f32(float x) {
    if (x>20.0f) return x;
    if (x< -20.0f) return expf(x);
    return log1pf(expf(x));
}
