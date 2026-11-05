#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
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
static uint32_t state = 0x7452ef09u;
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
        fprintf(stderr, "case %s index %zu expected %.9g got %.9g\n", n, i, (double)e, (double)g);
        return 1;
    } return 0;
}
static int      verify(const char *name, float *dst, const float *src, size_t n, float f){
    float          *old = malloc(n * sizeof *old);
    if (!old)
        return 2;
    for (size_t i = 0; i < n; i++)
        old[i] = src[i];
    ICASE(name, 0, 0, scale_alias_span(dst, src, n, f));
    for (size_t i = 0; i < n; i++)
        if (fcase(name, i, old[i] * f, dst[i]))
            return 1;
    free(old);
    return 0;
}
int
main(void)
{
    {
        float           src[] = {1, -2, 3.5f, 4}, dst[4] = {0};
        if (verify("disjoint-scale", dst, src, 4, 2) || verify("zero-factor", dst, src, 4, 0))
            return 1;
        float           same[] = {1, -2, 3.5f, 4};
        if (verify("exact-in-place", same, same, 4, -0.5f))
            return 1;
    }
    {
        float           a[8] = {1, 2, 3, 4, 5, 6, 7, 8}, copy[8];
        memcpy(copy, a, sizeof a);
        ICASE("null-destination", 0, -1, scale_alias_span(NULL, a, 4, 2));
        ICASE("null-source", 0, -1, scale_alias_span(a, NULL, 4, 2));
        ICASE("zero-length", 0, -1, scale_alias_span(a, a, 0, 2));
        ICASE("huge-byte-count", 0, -1, scale_alias_span(a, a, SIZE_MAX, 2));
        ICASE("null-and-length-errors-atomic", 0, 1, a[0]);
        ICASE("null-and-length-errors-atomic", 7, 8, a[7]);
        ICASE("partial-overlap-forward", 0, -1, scale_alias_span(a + 1, a, 4, 2));
        for (size_t i = 0; i < 8; i++)
            if (fcase("partial-overlap-forward-atomic", i, copy[i], a[i]))
                return 1;
        ICASE("partial-overlap-reverse", 0, -1, scale_alias_span(a, a + 1, 4, 2));
        for (size_t i = 0; i < 8; i++)
            if (fcase("partial-overlap-reverse-atomic", i, copy[i], a[i]))
                return 1;
    }
    for (size_t c = 0; c < 240; c++) {
        size_t          n = 1 + rnd() % 257;
        float          *src = malloc(n * sizeof *src), *dst = malloc(n * sizeof *dst);
        if (!src || !dst)
            return 2;
        for (size_t i = 0; i < n; i++)
            src[i] = ((int)(rnd() % 4001) - 2000) / 128.0f;
        float           factor = ((int)(rnd() % 2001) - 1000) / 333.0f;
        if (verify("random-disjoint-scale", dst, src, n, factor))
            return 1;
        if (c & 1u) {
            memcpy(dst, src, n * sizeof *src);
            if (verify("random-exact-alias", dst, dst, n, factor))
                return 1;
        } free(src);
        free(dst);
    }
    {
        size_t          n = 131071;
        float          *src = malloc(n * sizeof *src), *dst = malloc(n * sizeof *dst);
        if (!src || !dst)
            return 2;
        for (size_t i = 0; i < n; i++)
            src[i] = (float)((int)(i % 4001) - 2000) / 7.0f;
        if (verify("largest-disjoint-span", dst, src, n, 1.25f))
            return 1;
        free(src);
        free(dst);
    }
    if (tracked_heap_allocations == SIZE_MAX) {
        heldout_test_malloc(1);
        heldout_test_calloc(1, 1);
        heldout_test_realloc(NULL, 1);
    }
    ICASE("no-heap-allocation", 0, 0, tracked_heap_allocations);
    return 0;
}
