#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <float.h>
#include <math.h>
#include <string.h>
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
static uint32_t state = 0x1554dd31u;
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
static int      verify(const char *name, const float *x, float *y, int n){
    float          *src = malloc((size_t) n * sizeof *src);
    if (!src)
        return 2;
    for (int i = 0; i < n; i++)
        src[i] = x[i];
    double          m = src[0], sum = 0;
    for (int i = 1; i < n; i++)
        if (src[i] > m)
            m = src[i];
    for (int i = 0; i < n; i++)
        sum += exp((double)src[i] - m);
    ICASE(name, 0, 0, normalize_logits_unit(x, y, n));
    for (int i = 0; i < n; i++)
        if (fcase(name, (size_t) i, (float)(exp((double)src[i] - m) / sum), y[i]))
            return 1;
    free(src);
    return 0;
}
int
main(void)
{
    {
        float           x[] = {1000, 1001, 999}, y[3];
        if (verify("large-finite-logits", x, y, 3))
            return 1;
        float           same[] = {0, 0, 0, 0};
        if (verify("uniform-values", same, same, 4))
            return 1;
        float           gap[] = {-FLT_MAX, FLT_MAX}, out[2];
        if (verify("extreme-finite-gap", gap, out, 2))
            return 1;
    }
    {
        float           x[4] = {1, 2, 3, 4}, y[4] = {61, 62, 63, 64};
        ICASE("null-input", 0, -1, normalize_logits_unit(NULL, y, 4));
        ICASE("null-output", 0, -1, normalize_logits_unit(x, NULL, 4));
        ICASE("null-atomic", 0, 61, y[0]);
        ICASE("zero-n", 0, -1, normalize_logits_unit(x, y, 0));
        ICASE("negative-n", 0, -1, normalize_logits_unit(x, y, -1));
        ICASE("invalid-size-atomic", 0, 61, y[0]);
        float           nanx[] = {1, 2, NAN, 4}, infx[] = {1, 2, INFINITY, 4};
        ICASE("nan-input", 0, -1, normalize_logits_unit(nanx, y, 4));
        ICASE("infinite-input", 0, -1, normalize_logits_unit(infx, y, 4));
        ICASE("nonfinite-atomic", 0, 61, y[0]);
        ICASE("nonfinite-atomic", 3, 64, y[3]);
        float           a[6] = {1, 2, 3, 4, 5, 6}, copy[6];
        memcpy(copy, a, sizeof a);
        ICASE("partial-overlap-forward", 0, -1, normalize_logits_unit(a, a + 1, 4));
        ICASE("partial-overlap-forward-atomic", 0, 1, a[0]);
        ICASE("partial-overlap-forward-atomic", 5, 6, a[5]);
        ICASE("partial-overlap-reverse", 0, -1, normalize_logits_unit(a + 1, a, 4));
        for (size_t i = 0; i < 6; i++)
            if (fcase("partial-overlap-reverse-atomic", i, copy[i], a[i]))
                return 1;
    }
    for (size_t c = 0; c < 240; c++) {
        int             n = (int[]){1, 2, 3, 7, 8, 9, 15, 16, 17}[c % 9] + (int)(rnd() % 64);
        float          *x = malloc((size_t) n * sizeof *x), *y = malloc((size_t) n * sizeof *y);
        if (!x || !y)
            return 2;
        for (int i = 0; i < n; i++)
            x[i] = ((int)(rnd() % 20001) - 10000) / 128.0f;
        if (c & 1u) {
            x[0] = ((int)(rnd() % 1000) - 500) * 100.0f;
        } if (verify("random-softmax-property", x, y, n))
            return 1;
        free(x);
        free(y);
    }
    {
        int             n = 131071;
        float          *x = malloc((size_t) n * sizeof *x), *y = malloc((size_t) n * sizeof *y);
        if (!x || !y)
            return 2;
        for (int i = 0; i < n; i++)
            x[i] = (float)((int)(i % 997) - 498) / 11.0f;
        x[n - 1] = 100.0f;
        if (verify("largest-softmax-vector", x, y, n))
            return 1;
        free(x);
        free(y);
    }
    if (tracked_heap_allocations == SIZE_MAX) {
        heldout_test_malloc(1);
        heldout_test_calloc(1, 1);
        heldout_test_realloc(NULL, 1);
    }
    ICASE("no-heap-allocation", 0, 0, tracked_heap_allocations);
    return 0;
}
