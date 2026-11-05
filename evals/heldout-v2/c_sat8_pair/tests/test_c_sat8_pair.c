#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
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
static uint32_t state = 0x5a89d277u;
static uint32_t rnd(void){
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
#define ICASE(name,idx,exp,got) do{long long e_=(long long)(exp),g_=(long long)(got);if(e_!=g_){fprintf(stderr,"case %s index %zu expected %lld got %lld\n",name,(size_t)(idx),e_,g_);return 1;}}while(0)
static int      verify(const char *name, const uint8_t * a, const uint8_t * b, uint8_t * d, size_t n, uint8_t cap){
    ICASE(name, 0, 0, saturating_pair_u8(d, a, b, n, cap));
    for (size_t i = 0; i < n; i++) {
        unsigned        sum = (unsigned)a[i] + (unsigned)b[i];
        unsigned        e = sum > cap ? cap : sum;
        ICASE(name, i, e, d[i]);
    } return 0;
}
int
main(void)
{
    {
        uint8_t         a[] = {250, 10, 4, 255}, b[] = {10, 20, 9, 1}, d[4] = {7, 7, 7, 7};
        if (verify("saturation-ceiling", a, b, d, 4, 255) || verify("custom-ceiling", a, b, d, 4, 17) || verify("zero-ceiling", a, b, d, 4, 0))
            return 1;
    }
    {
        uint8_t         a[3] = {250, 8, 1}, b[3] = {20, 9, 2}, d[3] = {61, 62, 63};
        ICASE("null-destination", 0, -1, saturating_pair_u8(NULL, a, b, 3, 20));
        ICASE("null-a", 0, -1, saturating_pair_u8(d, NULL, b, 3, 20));
        ICASE("null-b", 0, -1, saturating_pair_u8(d, a, NULL, 3, 20));
        ICASE("null-errors-atomic", 0, 61, d[0]);
        ICASE("null-errors-atomic", 1, 62, d[1]);
        ICASE("zero-length", 0, -1, saturating_pair_u8(d, a, b, 0, 20));
        ICASE("zero-length-atomic", 0, 61, d[0]);
    }
    for (size_t c = 0; c < 240; c++) {
        size_t          n = 1 + rnd() % 513;
        uint8_t        *a = malloc(n), *b = malloc(n), *d = malloc(n);
        if (!a || !b || !d)
            return 2;
        uint8_t         cap = (uint8_t) rnd();
        for (size_t i = 0; i < n; i++) {
            a[i] = (uint8_t) rnd();
            b[i] = (uint8_t) rnd();
        } if (verify("random-unsigned-saturation", a, b, d, n, cap))
            return 1;
        free(a);
        free(b);
        free(d);
    }
    {
        size_t          n = 131071;
        uint8_t        *a = malloc(n), *b = malloc(n), *d = malloc(n);
        if (!a || !b || !d)
            return 2;
        for (size_t i = 0; i < n; i++) {
            a[i] = (uint8_t) (i * 37u);
            b[i] = (uint8_t) (255u - i * 19u);
        } if (verify("largest-odd-length", a, b, d, n, 173))
            return 1;
        free(a);
        free(b);
        free(d);
    }
    if (tracked_heap_allocations == SIZE_MAX) {
        heldout_test_malloc(1);
        heldout_test_calloc(1, 1);
        heldout_test_realloc(NULL, 1);
    }
    ICASE("no-heap-allocation", 0, 0, tracked_heap_allocations);
    return 0;
}
