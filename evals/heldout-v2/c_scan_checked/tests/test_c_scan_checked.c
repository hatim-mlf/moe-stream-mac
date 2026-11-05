#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
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
static uint32_t state = 0x08d3ab2fu;
static uint32_t rnd(void){
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
#define ICASE(name,idx,exp,got) do{long long e_=(long long)(exp),g_=(long long)(got);if(e_!=g_){fprintf(stderr,"case %s index %zu expected %lld got %lld\n",name,(size_t)(idx),e_,g_);return 1;}}while(0)
static int      verify(const char *name, const int32_t * in, int32_t * out, size_t n){
    int64_t         sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += in[i];
        if (sum > INT32_MAX || sum < INT32_MIN) {
            fprintf(stderr, "test oracle overflow in %s at %zu\n", name, i);
            return 2;
        }
    } int           rc = checked_prefix_i32(in, out, n);
    ICASE(name, 0, 0, rc);
    sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += in[i];
        ICASE(name, i, sum, out[i]);
    } return 0;
}
int
main(void)
{
    {
        int32_t         a[] = {3, -1, 4, 2}, out[4] = {71, 72, 73, 74};
        if (verify("inclusive-scan", a, out, 4))
            return 1;
        out[0] = 71;
        out[1] = 72;
        out[2] = 73;
        out[3] = 74;
        ICASE("null-input", 0, -1, checked_prefix_i32(NULL, out, 4));
        ICASE("null-output", 0, -1, checked_prefix_i32(a, NULL, 4));
        ICASE("null-errors-atomic", 0, 71, out[0]);
        ICASE("null-errors-atomic", 3, 74, out[3]);
        ICASE("zero-length", 0, -1, checked_prefix_i32(a, out, 0));
        ICASE("zero-length-atomic", 0, 71, out[0]);
    }
    {
        int32_t         hi[] = {INT32_MAX, 1}, lo[] = {INT32_MIN, -1}, late_hi[] = {1, 2, INT32_MAX}, late_lo[] = {-1, -2, INT32_MIN}, out[3] = {51, 52, 53};
        ICASE("positive-overflow", 0, -2, checked_prefix_i32(hi, out, 2));
        ICASE("positive-overflow-atomic", 0, 51, out[0]);
        ICASE("positive-overflow-atomic", 1, 52, out[1]);
        ICASE("negative-overflow", 0, -2, checked_prefix_i32(lo, out, 2));
        ICASE("negative-overflow-atomic", 0, 51, out[0]);
        ICASE("late-positive-overflow", 0, -2, checked_prefix_i32(late_hi, out, 3));
        ICASE("late-overflow-atomic", 0, 51, out[0]);
        ICASE("late-overflow-atomic", 1, 52, out[1]);
        ICASE("late-overflow-atomic", 2, 53, out[2]);
        ICASE("late-negative-overflow", 0, -2, checked_prefix_i32(late_lo, out, 3));
        ICASE("late-negative-atomic", 0, 51, out[0]);
    }
    {
        int32_t         b[] = {INT32_MIN, 0, INT32_MAX}, out[3] = {0};
        if (verify("inclusive-int32-boundaries", b, out, 3))
            return 1;
    }
    for (size_t c = 0; c < 240; c++) {
        size_t          n = 1 + rnd() % 257;
        int32_t        *in = malloc(n * sizeof *in), *out = malloc(n * sizeof *out);
        if (!in || !out)
            return 2;
        for (size_t i = 0; i < n; i++)
            in[i] = (int32_t) (rnd() % 20001) - 10000;
        if (verify("random-prefix-property", in, out, n))
            return 1;
        free(in);
        free(out);
    }
    {
        size_t          n = 131071;
        int32_t        *in = malloc(n * sizeof *in), *out = malloc(n * sizeof *out);
        if (!in || !out)
            return 2;
        for (size_t i = 0; i < n; i++)
            in[i] = 1;
        if (verify("largest-nonmultiple", in, out, n))
            return 1;
        free(in);
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
