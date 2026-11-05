#include <stddef.h>
float strict_upper_sum_f32(const float *matrix, size_t n) {
    float sum = 0.0f;
    for (size_t r = 0; r < n; ++r)
        for (size_t c = r + 1; c < n; ++c) sum += matrix[r * n + c];
    return sum;
}
