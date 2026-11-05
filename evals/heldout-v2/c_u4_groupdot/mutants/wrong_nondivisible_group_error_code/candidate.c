#include <stdint.h>
int affine_nibble_project(const uint8_t *packed, const float *scales, const float *biases,
                          const float *x, int n, int group, float *out) {
    int g, j;
    float total = 0.0f;
    if (packed == 0 || scales == 0 || biases == 0 || x == 0 || out == 0 || n <= 0 ||
        group <= 0) return -1;
    if (n % group != 0) return -2;
    for (g = 0; g < n / group; ++g) {
        float sum_q = 0.0f, sum_x = 0.0f;
        for (j = 0; j < group; ++j) {
            int i = g * group + j;
            unsigned q = (unsigned)((packed[i / 2] >> ((i % 2) * 4)) & 15u);
            sum_q += (float)q * x[i];
            sum_x += x[i];
        }
        total += scales[g] * sum_q + biases[g] * sum_x;
    }
    *out = total;
    return 0;
}
