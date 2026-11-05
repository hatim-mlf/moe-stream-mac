#include <math.h>
float deadzone_f32(float x, float threshold) {
    if (!isfinite(x) || !isfinite(threshold) || threshold < 0.0f) return 0.0f;
    if (fabsf(x) <= threshold) return 0.0f;
    return x > 0.0f ? x - threshold : x + threshold;
}
