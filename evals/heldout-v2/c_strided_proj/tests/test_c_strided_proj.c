#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
static size_t tracked_heap_allocations;
static void *heldout_test_malloc(size_t size);
static void *heldout_test_calloc(size_t count, size_t size);
static void *heldout_test_realloc(void *ptr, size_t size);
#define malloc heldout_test_malloc
#define calloc heldout_test_calloc
#define realloc heldout_test_realloc
#include "../candidate.c"
#undef malloc
#undef calloc
#undef realloc
static void *heldout_test_malloc(size_t size) { tracked_heap_allocations++; return malloc(size); }
static void *heldout_test_calloc(size_t count, size_t size) { tracked_heap_allocations++; return calloc(count, size); }
static void *heldout_test_realloc(void *ptr, size_t size) { tracked_heap_allocations++; return realloc(ptr, size); }
static uint32_t state = 0x9e014671u;
static uint32_t rnd(void){
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
#define ICASE(name,idx,exp,got) do{long long e_=(long long)(exp),g_=(long long)(got);if(e_!=g_){fprintf(stderr,"case %s index %zu expected %lld got %lld\n",name,(size_t)(idx),e_,g_);return 1;}}while(0)
/* Float comparisons use relative tolerance 1e-5 plus absolute tolerance 1e-6. */
static int      fcase(const char *n, size_t i, float e, float g){
    float           tol = 1e-6f + 1e-5f * fabsf(e);
    if (!(fabsf(e - g) <= tol)) {
        fprintf(stderr, "case %s index %zu expected %.9g got %.9g (tol %.3g)\n", n, i, (double)e, (double)g, (double)tol);
        return 1;
    } return 0;
}
static int      verify(const char *name, const float *a, const float *x, int rows, int cols, ptrdiff_t rs, ptrdiff_t cs, ptrdiff_t xs, float *out){
    int             rc = project_strided_grid(a, x, rows, cols, rs, cs, xs, out);
    ICASE(name, 0, 0, rc);
    for (int r = 0; r < rows; r++) {
        double          sum = 0;
        for (int c = 0; c < cols; c++)
            sum += (double)a[(ptrdiff_t) r * rs + (ptrdiff_t) c * cs] * x[(ptrdiff_t) c * xs];
        if (fcase(name, (size_t) r, (float)sum, out[r]))
            return 1;
    } return 0;
}
int
main(void)
{
    {
        float           a[] = {1, 4, 2, 5, 3, 6}, x[] = {1, 99, 2, 99, 3}, out[2] = {51, 52};
        if (verify("transposed-interleaved-strides", a, x, 2, 3, 1, 2, 2, out))
            return 1;
    }
    {
        float           a[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, x[8] = {1, 2, 3, 4, 5, 6, 7, 8}, out[4] = {71, 72, 73, 74};
        ICASE("null-matrix", 0, -1, project_strided_grid(NULL, x, 2, 2, 2, 1, 1, out));
        ICASE("null-vector", 0, -1, project_strided_grid(a, NULL, 2, 2, 2, 1, 1, out));
        ICASE("null-output", 0, -1, project_strided_grid(a, x, 2, 2, 2, 1, 1, NULL));
        ICASE("null-atomic", 0, 71, out[0]);
        ICASE("null-atomic", 3, 74, out[3]);
        ICASE("zero-rows", 0, -1, project_strided_grid(a, x, 0, 2, 2, 1, 1, out));
        ICASE("negative-rows", 0, -1, project_strided_grid(a, x, -1, 2, 2, 1, 1, out));
        ICASE("zero-cols", 0, -1, project_strided_grid(a, x, 2, 0, 2, 1, 1, out));
        ICASE("negative-cols", 0, -1, project_strided_grid(a, x, 2, -1, 2, 1, 1, out));
        ICASE("zero-row-stride", 0, -1, project_strided_grid(a, x, 2, 2, 0, 1, 1, out));
        ICASE("negative-row-stride", 0, -1, project_strided_grid(a, x, 2, 2, -1, 1, 1, out));
        ICASE("zero-col-stride", 0, -1, project_strided_grid(a, x, 2, 2, 2, 0, 1, out));
        ICASE("negative-col-stride", 0, -1, project_strided_grid(a, x, 2, 2, 2, -1, 1, out));
        ICASE("zero-vector-stride", 0, -1, project_strided_grid(a, x, 2, 2, 2, 1, 0, out));
        ICASE("negative-vector-stride", 0, -1, project_strided_grid(a, x, 2, 2, 2, 1, -1, out));
        ICASE("invalid-atomic", 0, 71, out[0]);
        ICASE("invalid-atomic", 3, 74, out[3]);
    }
    for (size_t c = 0; c < 240; c++) {
        int             rows = 1 + (int)(rnd() % 15), cols = 1 + (int)(rnd() % 19);
        ptrdiff_t       cs = 1 + (ptrdiff_t) (rnd() % 4), rs = (ptrdiff_t) cols * cs + 1 + (ptrdiff_t) (rnd() % 5),
                        xs = 1 + (ptrdiff_t) (rnd() % 4);
        size_t          an = (size_t) ((rows - 1) * rs + (cols - 1) * cs + 1), xn = (size_t) ((cols - 1) * xs + 1);
        float          *a = malloc(an * sizeof *a), *x = malloc(xn * sizeof *x), *out = malloc((size_t) rows * sizeof *out);
        if (!a || !x || !out)
            return 2;
        for (size_t i = 0; i < an; i++)
            a[i] = ((int)(rnd() % 2001) - 1000) / 100.0f;
        for (size_t i = 0; i < xn; i++)
            x[i] = ((int)(rnd() % 2001) - 1000) / 200.0f;
        if (verify("random-positive-strides", a, x, rows, cols, rs, cs, xs, out))
            return 1;
        free(a);
        free(x);
        free(out);
    }
    {
        int             rows = 257, cols = 263;
        ptrdiff_t       rs = 1, cs = rows + 2, xs = 3;
        size_t          an = (size_t) ((cols - 1) * cs + rows), xn = (size_t) ((cols - 1) * xs + 1);
        float          *a = malloc(an * sizeof *a), *x = malloc(xn * sizeof *x), *out = malloc((size_t) rows * sizeof *out);
        if (!a || !x || !out)
            return 2;
        for (size_t i = 0; i < an; i++)
            a[i] = (float)((int)(i % 101) - 50) / 10.0f;
        for (size_t i = 0; i < xn; i++)
            x[i] = (float)((int)(i % 31) - 15) / 7.0f;
        if (verify("largest-transposed-grid", a, x, rows, cols, rs, cs, xs, out))
            return 1;
        free(a);
        free(x);
        free(out);
    }
    if (tracked_heap_allocations == SIZE_MAX) {
        heldout_test_malloc(1);
        heldout_test_calloc(1, 1);
        heldout_test_realloc(NULL, 1);
    }
    ICASE("no-heap-allocation", 0, 0, tracked_heap_allocations);
    return 0;
}
