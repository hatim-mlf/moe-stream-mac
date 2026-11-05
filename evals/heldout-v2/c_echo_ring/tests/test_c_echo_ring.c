#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
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
static uint32_t state = 0x60a9d771u;
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
        TraceRing       r;
        int             storage[3] = {0}, out = 77;
        ICASE("init-null-ring", 0, -1, trail_ring_init(NULL, storage, 3));
        ICASE("init-null-storage", 0, -1, trail_ring_init(&r, NULL, 3));
        ICASE("init-zero-capacity", 0, -1, trail_ring_init(&r, storage, 0));
        ICASE("init-success", 0, 0, trail_ring_init(&r, storage, 3));
        ICASE("init-empty-size", 0, 0, r.size);
        ICASE("empty-get-age", 0, -1, trail_ring_get(&r, 0, &out));
        ICASE("empty-get-atomic", 0, 77, out);
        ICASE("get-null-ring", 0, -1, trail_ring_get(NULL, 0, &out));
        ICASE("get-null-output", 0, -1, trail_ring_get(&r, 0, NULL));
        ICASE("push-null-ring", 0, -1, trail_ring_push(NULL, 1));
        ICASE("push-success", 0, 0, trail_ring_push(&r, 10));
        ICASE("push-success", 1, 0, trail_ring_push(&r, 20));
        ICASE("push-success", 2, 0, trail_ring_push(&r, 30));
        ICASE("push-wrap", 3, 0, trail_ring_push(&r, 40));
        ICASE("get-newest", 0, 0, trail_ring_get(&r, 0, &out));
        ICASE("get-newest", 1, 40, out);
        ICASE("get-oldest", 0, 0, trail_ring_get(&r, 2, &out));
        ICASE("get-oldest", 1, 20, out);
        out = 91;
        ICASE("unavailable-age", 0, -1, trail_ring_get(&r, 3, &out));
        ICASE("unavailable-age-atomic", 0, 91, out);
        TraceRing       bad = r;
        bad.data = NULL;
        ICASE("invalid-null-data-get", 0, -1, trail_ring_get(&bad, 0, &out));
        bad = r;
        bad.capacity = 0;
        ICASE("invalid-zero-capacity-get", 0, -1, trail_ring_get(&bad, 0, &out));
        bad = r;
        bad.next = bad.capacity;
        ICASE("invalid-next-get", 0, -1, trail_ring_get(&bad, 0, &out));
        bad = r;
        bad.size = bad.capacity + 1;
        ICASE("invalid-size-get", 0, -1, trail_ring_get(&bad, 0, &out));
        bad = r;
        bad.next = bad.capacity;
        ICASE("invalid-next-push", 0, -1, trail_ring_push(&bad, 88));
        bad = r;
        bad.size = bad.capacity + 1;
        ICASE("invalid-size-push", 0, -1, trail_ring_push(&bad, 88));
        bad = r;
        bad.data = NULL;
        ICASE("invalid-null-data-push", 0, -1, trail_ring_push(&bad, 88));
        bad = r;
        bad.capacity = 0;
        ICASE("invalid-zero-capacity-push", 0, -1, trail_ring_push(&bad, 88));
    }
    for (size_t c = 0; c < 256; c++) {
        size_t          cap = 1 + rnd() % 64;
        int             store[64], model[64], size = 0;
        TraceRing       r;
        ICASE("random-ring-init", c, 0, trail_ring_init(&r, store, cap));
        for (size_t j = 0; j < 200; j++) {
            int             value = (int)(rnd() % 200001) - 100000;
            ICASE("random-ring-push", j, 0, trail_ring_push(&r, value));
            if ((size_t) size < cap)
                model[size++] = value;
            else {
                for (size_t k = 1; k < cap; k++)
                    model[k - 1] = model[k];
                model[cap - 1] = value;
            } ICASE("random-ring-size", j, size, r.size);
            for (int age = 0; age < size; age++) {
                int             got = -12345;
                ICASE("random-ring-get-rc", j, 0, trail_ring_get(&r, (size_t) age, &got));
                ICASE("random-ring-get", (size_t) age, model[size - 1 - age], got);
            }
        }
    }
    {
        static int      storage[4096];
        TraceRing       r;
        ICASE("large-capacity-init", 0, 0, trail_ring_init(&r, storage, 4096));
        for (int i = 0; i < 65537; i++)
            ICASE("large-wrap-push", (size_t) i, 0, trail_ring_push(&r, i));
        int             got = -1;
        ICASE("large-wrap-newest-rc", 0, 0, trail_ring_get(&r, 0, &got));
        ICASE("large-wrap-newest", 0, 65536, got);
        ICASE("large-wrap-oldest-rc", 0, 0, trail_ring_get(&r, 4095, &got));
        ICASE("large-wrap-oldest", 0, 61441, got);
    }
    if (tracked_heap_allocations == SIZE_MAX) {
        heldout_test_malloc(1);
        heldout_test_calloc(1, 1);
        heldout_test_realloc(NULL, 1);
    }
    ICASE("no-heap-allocation", 0, 0, tracked_heap_allocations);
    return 0;
}
