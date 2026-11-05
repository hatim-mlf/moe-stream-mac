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
static uint32_t state = 0x3b17a55du;
static uint32_t rnd(void){
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
#define ICASE(name,idx,exp,got) do{long long e_=(long long)(exp),g_=(long long)(got);if(e_!=g_){fprintf(stderr,"case %s index %zu expected %lld got %lld\n",name,(size_t)(idx),e_,g_);return 1;}}while(0)
int
main(void)
{
    {
        uint8_t         src[] = {0x71, 0x01}, out[4] = {99, 99, 99, 99};
        int             rc = unpack_threebit_stream(src, 2, 3, out);
        ICASE("cross-byte-values", 0, 0, rc);
        for (size_t i = 0; i < 3; i++) {
            unsigned        e[] = {1, 6, 5};
            ICASE("cross-byte-values", i, e[i], out[i]);
        }
    }
    {
        uint8_t         src[3] = {0xFF, 0xFF, 0xFF}, out[8];
        memset(out, 0xA5, sizeof out);
        ICASE("zero-count-null-pointers", 0, 0, unpack_threebit_stream(NULL, 0, 0, NULL));
        ICASE("zero-count-nonnull-pointers", 0, 0, unpack_threebit_stream(src, 0, 0, out));
        ICASE("zero-count-no-write", 0, 0xA5, out[0]);
    }
    {
        uint8_t         src[2] = {0x71, 0x01}, out[4] = {7, 8, 9, 10};
        int             expected[4] = {7, 8, 9, 10};
        ICASE("short-source", 0, -1, unpack_threebit_stream(src, 1, 3, out));
        for (size_t i = 0; i < 4; i++)
            ICASE("short-source-atomic", i, expected[i], out[i]);
        ICASE("null-source", 0, -1, unpack_threebit_stream(NULL, 2, 3, out));
        ICASE("null-destination", 0, -1, unpack_threebit_stream(src, 2, 3, NULL));
        ICASE("null-source-atomic", 0, 7, out[0]);
        ICASE("count-arithmetic-overflow", 0, -1, unpack_threebit_stream(src, SIZE_MAX, SIZE_MAX, out));
        ICASE("count-overflow-atomic", 0, 7, out[0]);
    }
    for (size_t c = 0; c < 256; c++) {
        size_t          n = c < 16 ? (size_t[]) {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 47}[c % 16] : 1 + rnd() % 513;
        size_t          bytes = (3 * n + 7) / 8;
        uint8_t         src[200], got[600], expected[600];
        memset(src, 0, sizeof src);
        memset(got, 0xCC, sizeof got);
        for (size_t i = 0; i < n; i++) {
            unsigned        q = rnd() % 8;
            for (unsigned b = 0; b < 3; b++) {
                if ((q >> b) & 1u) {
                    size_t bit = 3 * i + b;
                    src[bit / 8] |= (uint8_t)(1u << (bit % 8));
                }
            }
            expected[i] = (uint8_t)q;
        } int           rc = unpack_threebit_stream(src, bytes, n, got);
        ICASE("random-packed-stream", c, 0, rc);
        for (size_t i = 0; i < n; i++)
            ICASE("random-packed-stream", i, expected[i], got[i]);
    }
    {
        enum {
        N = 131071, B = (3 * N + 7) / 8};
        static uint8_t src[B], out[N];
        for (size_t i = 0; i < B; i++)
            src[i] = (uint8_t) rnd();
        ICASE("large-odd-count", 0, 0, unpack_threebit_stream(src, B, N, out));
        for (size_t i = 0; i < N; i++) {
            size_t          bit = 3 * i;
            unsigned        word = src[bit / 8];
            if (bit % 8 > 5)
                word |= (unsigned)src[bit / 8 + 1] << 8;
            ICASE("large-odd-count", i, (word >> (bit % 8)) & 7u, out[i]);
        }
    }
    if (tracked_heap_allocations == SIZE_MAX) {
        heldout_test_malloc(1);
        heldout_test_calloc(1, 1);
        heldout_test_realloc(NULL, 1);
    }
    ICASE("no-heap-allocation", 0, 0, tracked_heap_allocations);
    return 0;
}
