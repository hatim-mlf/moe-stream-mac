#include <math.h>
int normalize_quat_f32(const float in[4], float out[4]) {
    if (!in || !out) return -1;
    double sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        if (!isfinite(in[i])) return -1;
        sum += (double)in[i] * in[i];
    }
    if (sum == 0.0) return -1;
    double inv = 1.0 / sqrt(sum);
    float temp[4];
    for (int i = 0; i < 4; ++i) temp[i] = (float)((double)in[i] * inv);
    for (int i = 0; i < 4; ++i) out[i] = temp[i];
    return 0;
}
