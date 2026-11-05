#include <stddef.h>
void max_pool2x2_f32(const float *x, size_t rows, size_t cols, float *out) {
    size_t out_rows = rows / 2, out_cols = cols / 2;
    for (size_t r = 0; r < out_rows; ++r) for (size_t c = 0; c < out_cols; ++c) {
        size_t i = (2*r) * cols + 2*c;
        float m = x[i];
        if (x[i+1] > m) m = x[i+1];
        if (x[i+cols] > m) m = x[i+cols];
        if (x[i+cols+1] > m) m = x[i+cols+1];
        out[r*out_cols+c] = m;
    }
}
