#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <float.h>
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
static uint32_t state = 0x2cb7e981u;
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
static int      verify(const char *name, const float *x, const float *r, const float *g, int n, float eps, float *out){
    float           sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float           z = x[i] + r[i];
        sum += z * z;
    } float         den = sqrtf(sum / (float)n + eps);
    int             rc = rms_gate_vector(x, r, g, n, eps, out);
    ICASE(name, 0, 0, rc);
    for (int i = 0; i < n; i++)
        if (fcase(name, (size_t) i, ((x[i] + r[i]) / den) * g[i], out[i]))
            return 1;
    return 0;
}
int
main(void)
{
    {
        float           x[] = {3, 0}, r[] = {1, 0}, g[] = {2, 1}, out[] = {77, 88};
        if (verify("residual-rms-gate", x, r, g, 2, 1, out))
            return 1;
    }
    {
        float           x[2] = {1, 2}, r[2] = {0, 0}, g[2] = {1, 1}, out[2] = {71, 72};
        ICASE("null-x", 0, -1, rms_gate_vector(NULL, r, g, 2, 1, out));
        ICASE("null-residual", 0, -1, rms_gate_vector(x, NULL, g, 2, 1, out));
        ICASE("null-gate", 0, -1, rms_gate_vector(x, r, NULL, 2, 1, out));
        ICASE("null-output", 0, -1, rms_gate_vector(x, r, g, 2, 1, NULL));
        ICASE("null-errors-atomic", 0, 71, out[0]);
        ICASE("null-errors-atomic", 1, 72, out[1]);
        ICASE("zero-n", 0, -1, rms_gate_vector(x, r, g, 0, 1, out));
        ICASE("negative-n", 0, -1, rms_gate_vector(x, r, g, -1, 1, out));
        ICASE("zero-epsilon", 0, -1, rms_gate_vector(x, r, g, 2, 0, out));
        ICASE("negative-epsilon", 0, -1, rms_gate_vector(x, r, g, 2, -1, out));
        ICASE("nan-epsilon", 0, -1, rms_gate_vector(x, r, g, 2, NAN, out));
        ICASE("infinite-epsilon", 0, -1, rms_gate_vector(x, r, g, 2, INFINITY, out));
        ICASE("epsilon-errors-atomic", 0, 71, out[0]);
        ICASE("epsilon-errors-atomic", 1, 72, out[1]);
        float           badx[] = {1, FLT_MAX}, badr[] = {0, FLT_MAX};
        ICASE("nonfinite-residual-sum", 0, -1, rms_gate_vector(badx, badr, g, 2, 1, out));
        float           badg[] = {1, INFINITY};
        ICASE("nonfinite-gate", 0, -1, rms_gate_vector(x, r, badg, 2, 1, out));
        ICASE("data-errors-atomic", 0, 71, out[0]);
        ICASE("data-errors-atomic", 1, 72, out[1]);
        float           nanx[] = {1, NAN};
        ICASE("nan-z", 0, -1, rms_gate_vector(nanx, r, g, 2, 1, out));
        float           nang[] = {1, NAN};
        ICASE("nan-gate", 0, -1, rms_gate_vector(x, r, nang, 2, 1, out));
        ICASE("nan-gate-atomic", 0, 71, out[0]);
    }
    for (size_t c = 0; c < 240; c++) {
        int             n = 1 + (int)(rnd() % 300);
        float          *x = malloc((size_t) n * sizeof *x), *r = malloc((size_t) n * sizeof *r),
                       *g = malloc((size_t) n * sizeof *g), *out = malloc((size_t) n * sizeof *out);
        if (!x || !r || !g || !out)
            return 2;
        for (int i = 0; i < n; i++) {
            x[i] = ((int)(rnd() % 20001) - 10000) / 256.0f;
            r[i] = ((int)(rnd() % 20001) - 10000) / 512.0f;
            g[i] = ((int)(rnd() % 20001) - 10000) / 1024.0f;
        } if (verify("random-rms-gate", x, r, g, n, 0.001f + (rnd() % 1000) / 1000.0f, out))
            return 1;
        free(x);
        free(r);
        free(g);
        free(out);
    }
    {
        int             n = 131071;
        float          *x = malloc((size_t) n * sizeof *x), *r = malloc((size_t) n * sizeof *r),
                       *g = malloc((size_t) n * sizeof *g), *out = malloc((size_t) n * sizeof *out);
        if (!x || !r || !g || !out)
            return 2;
        for (int i = 0; i < n; i++) {
            x[i] = (float)((i % 17) - 8) / 9.0f;
            r[i] = (float)((i % 11) - 5) / 13.0f;
            g[i] = (float)((i % 7) - 3) / 5.0f;
        } if (verify("largest-vector", x, r, g, n, 1e-4f, out))
            return 1;
        free(x);
        free(r);
        free(g);
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
