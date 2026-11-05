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
static uint32_t state = 0x6c5f349bu;
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
static double   naive_lse(const float *x, size_t n){
    double          m = x[0], s = 0;
    for (size_t i = 1; i < n; i++)
        if (x[i] > m)
            m = x[i];
    for (size_t i = 0; i < n; i++)
        s += exp((double)x[i] - m);
    return m + log(s);
}
int
main(void)
{
    {
        float           x[4] = {1000, 1000, -2, 0}, out[3] = {71, 72, 73};
        size_t          lens[2] = {2, 2};
        int             rc = group_log_partition(x, lens, 4, 2, out);
        ICASE("stable-groups-rc", 0, 0, rc);
        if (fcase("stable-groups", 0, (float)(1000 + log(2.0)), out[0]) || fcase("stable-groups", 1, (float)log(exp(-2.0) + 1.0), out[1]))
            return 1;
        out[0] = 71;
        out[1] = 72;
        lens[0] = 0;
        ICASE("zero-group-length", 0, -1, group_log_partition(x, lens, 4, 2, out));
        if (fcase("zero-group-atomic", 0, 71, out[0]) || fcase("zero-group-atomic", 1, 72, out[1]))
            return 1;
        lens[0] = 3;
        lens[1] = 2;
        ICASE("length-sum-too-large", 0, -1, group_log_partition(x, lens, 4, 2, out));
        lens[0] = 1;
        lens[1] = 1;
        ICASE("length-sum-too-small", 0, -1, group_log_partition(x, lens, 4, 2, out));
        lens[0] = 2;
        lens[1] = 2;
        ICASE("zero-total", 0, -1, group_log_partition(x, lens, 0, 2, out));
        ICASE("zero-groups", 0, -1, group_log_partition(x, lens, 4, 0, out));
        ICASE("null-x", 0, -1, group_log_partition(NULL, lens, 4, 2, out));
        ICASE("null-lengths", 0, -1, group_log_partition(x, NULL, 4, 2, out));
        ICASE("null-output", 0, -1, group_log_partition(x, lens, 4, 2, NULL));
        float           badx[4] = {1, 2, NAN, 4};
        ICASE("nonfinite-input", 0, -1, group_log_partition(badx, lens, 4, 2, out));
        if (fcase("nonfinite-input-atomic", 0, 71, out[0]) || fcase("nonfinite-input-atomic", 1, 72, out[1]))
            return 1;
    }
    for (size_t c = 0; c < 240; c++) {
        size_t          groups = 1 + rnd() % 12, total = 0, lens[12];
        for (size_t g = 0; g < groups; g++) {
            lens[g] = 1 + rnd() % 17;
            total += lens[g];
        } float        *x = malloc(total * sizeof *x), *out = malloc(groups * sizeof *out);
        if (!x || !out)
            return 2;
        for (size_t i = 0; i < total; i++)
            x[i] = ((int)(rnd() % 40001) - 20000) / 128.0f;
        int             rc = group_log_partition(x, lens, total, groups, out);
        ICASE("random-variable-groups", c, 0, rc);
        size_t          pos = 0;
        for (size_t g = 0; g < groups; g++) {
            float           e = (float)naive_lse(x + pos, lens[g]);
            if (fcase("random-variable-groups", g, e, out[g]))
                return 1;
            pos += lens[g];
        } free(x);
        free(out);
    }
    {
        size_t          total = 131071, groups = 257, lens[257];
        float          *x = malloc(total * sizeof *x), *out = malloc(groups * sizeof *out);
        if (!x || !out)
            return 2;
        size_t          pos = 0;
        for (size_t g = 0; g < groups; g++) {
            lens[g] = (g == groups - 1) ? total - pos : total / groups;
            pos += lens[g];
        } for (size_t i = 0; i < total; i++)
            x[i] = (float)((int)(i % 409) - 204) / 7.0f;
        ICASE("largest-uneven-total", 0, 0, group_log_partition(x, lens, total, groups, out));
        pos = 0;
        for (size_t g = 0; g < groups; g++) {
            if (fcase("largest-uneven-total", g, (float)naive_lse(x + pos, lens[g]), out[g]))
                return 1;
            pos += lens[g];
        } free(x);
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
