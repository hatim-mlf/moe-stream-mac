#include <stddef.h>
void matvec_rowmajor_f32(const float *matrix, const float *x, size_t rows, size_t cols, float *y) {
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (size_t c = 0; c < cols; c++) sum += matrix[r * cols + c] * x[c];
        y[r] = sum;
    }
}
