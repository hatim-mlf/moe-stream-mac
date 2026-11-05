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
static uint32_t state = 0x1f6b82adu;
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
static unsigned qvalue(const uint8_t * p, int i){
    return (unsigned)((p[(unsigned)i / 2] >> ((i % 2) * 4)) & 15u);
}
static float    oracle(const uint8_t * p, const float *s, const float *b, const float *x, int n, int group){
    float           total = 0;
    for (int g = 0; g < n / group; g++) {
        float           sq = 0, sx = 0;
        for (int j = 0; j < group; j++) {
            int             i = g * group + j;
            unsigned        q = qvalue(p, i);
            sq += (float)q * x[i];
            sx += x[i];
        } total += s[g] * sq + b[g] * sx;
    } return total;
}
static int      verify(const char *name, const uint8_t * p, const float *s, const float *b, const float *x, int n, int group){
    float           got = -713.25f, e = oracle(p, s, b, x, n, group);
    ICASE(name, 0, 0, affine_nibble_project(p, s, b, x, n, group, &got));
    return fcase(name, 0, e, got);
}
int
main(void)
{
    {
        uint8_t         p[] = {0xF1, 0xA3};
        float           s[] = {2, -1}, b[] = {0.5f, 1}, x[] = {1, 2, 3, 4};
        if (verify("unsigned-high-nibbles-multiple-groups", p, s, b, x, 4, 2))
            return 1;
    }
    {
        uint8_t         p[] = {0xF8, 0x07};
        float           s[4] = {1, 1, 1, 1}, b[4] = {0, 0, 0, 0}, x[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        if (verify("values-eight-through-fifteen", p, s, b, x, 8, 2))
            return 1;
    }
    {
        uint8_t         oddp[2] = {0x21, 0xF3};
        float           os[1] = {2}, ob[1] = {0.5f}, ox[3] = {1, 2, 3};
        if (verify("odd-count-ignores-unused-high-nibble", oddp, os, ob, ox, 3, 3))
            return 1;
    }
    {
        uint8_t         p[2] = {0x21, 0x43};
        float           s[2] = {1, 2}, b[2] = {3, 4}, x[4] = {1, 2, 3, 4};
        float           out = 77;
        ICASE("null-packed", 0, -1, affine_nibble_project(NULL, s, b, x, 4, 2, &out));
        ICASE("null-scales", 0, -1, affine_nibble_project(p, NULL, b, x, 4, 2, &out));
        ICASE("null-biases", 0, -1, affine_nibble_project(p, s, NULL, x, 4, 2, &out));
        ICASE("null-vector", 0, -1, affine_nibble_project(p, s, b, NULL, 4, 2, &out));
        ICASE("null-output", 0, -1, affine_nibble_project(p, s, b, x, 4, 2, NULL));
        ICASE("null-atomic", 0, 77, out);
        ICASE("zero-n", 0, -1, affine_nibble_project(p, s, b, x, 0, 2, &out));
        ICASE("negative-n", 0, -1, affine_nibble_project(p, s, b, x, -4, 2, &out));
        ICASE("zero-group", 0, -1, affine_nibble_project(p, s, b, x, 4, 0, &out));
        ICASE("negative-group", 0, -1, affine_nibble_project(p, s, b, x, 4, -2, &out));
        ICASE("nondivisible-group", 0, -1, affine_nibble_project(p, s, b, x, 3, 2, &out));
        ICASE("invalid-layout-atomic", 0, 77, out);
    }
    for (size_t c = 0; c < 240; c++) {
        int             groups = 1 + (int)(rnd() % 11), group = 1 + (int)(rnd() % 13), n = groups * group;
        size_t          nb = ((size_t) n + 1) / 2;
        uint8_t        *p = malloc(nb);
        float          *s = malloc((size_t) groups * sizeof *s), *b = malloc((size_t) groups * sizeof *b),
                       *x = malloc((size_t) n * sizeof *x);
        if (!p || !s || !b || !x)
            return 2;
        for (size_t i = 0; i < nb; i++)
            p[i] = (uint8_t) rnd();
        for (int g = 0; g < groups; g++) {
            s[g] = ((int)(rnd() % 2001) - 1000) / 113.0f;
            b[g] = ((int)(rnd() % 2001) - 1000) / 127.0f;
        } for (int i = 0; i < n; i++)
            x[i] = ((int)(rnd() % 2001) - 1000) / 211.0f;
        if (verify("random-packed-group-property", p, s, b, x, n, group))
            return 1;
        free(p);
        free(s);
        free(b);
        free(x);
    }
    {
        int             n = 131072, group = 1024, groups = n / group;
        size_t          nb = (size_t) n / 2;
        uint8_t        *p = malloc(nb);
        float          *s = malloc((size_t) groups * sizeof *s), *b = malloc((size_t) groups * sizeof *b),
                       *x = malloc((size_t) n * sizeof *x);
        if (!p || !s || !b || !x)
            return 2;
        for (size_t i = 0; i < nb; i++)
            p[i] = (uint8_t) (i * 83u);
        for (int i = 0; i < groups; i++) {
            s[i] = (i % 19 - 9) * 0.125f;
            b[i] = (i % 13 - 6) * 0.0625f;
        } for (int i = 0; i < n; i++)
            x[i] = (float)((i % 31) - 15) / 32.0f;
        if (verify("largest-packed-vector", p, s, b, x, n, group))
            return 1;
        free(p);
        free(s);
        free(b);
        free(x);
    }
    if (tracked_heap_allocations == SIZE_MAX) {
        heldout_test_malloc(1);
        heldout_test_calloc(1, 1);
        heldout_test_realloc(NULL, 1);
    }
    ICASE("no-heap-allocation", 0, 0, tracked_heap_allocations);
    return 0;
}
