```c
#include "ops.h"
#include <math.h>
#include <string.h>

void op_rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        sum_sq += (double)x[i] * x[i];
    }
    double rms = sqrt(sum_sq / n + eps);
    for (int i = 0; i < n; i++) {
        out[i] = x[i] / (float)rms * w[i];
    }
}

void op_silu_mul(float *out, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        out[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

void op_sigmoid(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

void op_softmax(float *out, const float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        out[i] = expf(x[i] - max_val);
        sum += out[i];
    }
    for (int i = 0; i < n; i++) {
        out[i] /= (float)sum;
    }
}

void op_topk(int *idx, const float *x, int n, int k) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        float val = x[i];
        int pos = count;
        // Find insertion point: strictly greater values move ahead
        while (pos > 0 && x[idx[pos - 1]] <= val) {
            pos--;
        }
        // Shift elements to make room
        if (pos < count) {
            memmove(idx + pos + 1, idx + pos, (count - pos) * sizeof(int));
        }
        idx[pos] = i;
        if (count < k) count++;
    }
}

void op_rope(float *x, int n_heads, int head_dim, int rot_dims, int pos, float theta) {
    int half_rot = rot_dims / 2;
    for (int h = 0; h < n_heads; h++) {
        float *head = x + h * head_dim;
        for (int i = 0; i < half_rot; i++) {
            double inv_freq = theta * pow(-0.5 * i / rot_dims, 1.0 / (rot_dims / 2.0));
            // Correction: inv_freq = theta ^ (-2i / rot_dims)
            double exponent = -2.0 * i / rot_dims;
            double inv_freq_corrected = pow(theta, exponent);
            double a = (double)pos * inv_freq_corrected;
            float cos_a = (float)cos(a);
            float sin_a = (float)sin(a);
            
            float x1 = head[i];
            float x2 = head[i + half_rot];
            
            head[i] = x1 * cos_a - x2 * sin_a;
            head[i + half_rot] = x2 * cos_a + x1 * sin_a;
        }
    }
}
```