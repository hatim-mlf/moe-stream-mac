#include <math.h>
int rank_largest_topk(const float *values, int n, int k, int *indices, float *ranked) {
    int i, j;
    if (values == 0 || indices == 0 || ranked == 0 || n <= 0 || k <= 0 || k > n) return -1;
    for (i = 0; i < n; ++i) if (!isfinite(values[i])) return -1;
    for (i = 0; i < k; ++i) { indices[i] = -1; ranked[i] = 0.0f; }
    for (i = 0; i < n; ++i) {
        int pos = k;
        for (j = 0; j < k; ++j) {
            if (indices[j] < 0 || values[i] >= ranked[j] ||
                (values[i] == ranked[j] && i < indices[j])) { pos = j; break; }
        }
        if (pos < k) {
            for (j = k - 1; j > pos; --j) { indices[j] = indices[j-1]; ranked[j] = ranked[j-1]; }
            indices[pos] = i; ranked[pos] = values[i];
        }
    }
    return 0;
}
