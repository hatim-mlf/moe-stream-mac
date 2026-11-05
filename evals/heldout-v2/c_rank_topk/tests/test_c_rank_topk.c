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
static uint32_t state = 0xe13a4f65u;
static uint32_t rnd(void){
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
#define ICASE(name,idx,exp,got) do{long long e_=(long long)(exp),g_=(long long)(got);if(e_!=g_){fprintf(stderr,"case %s index %zu expected %lld got %lld\n",name,(size_t)(idx),e_,g_);return 1;}}while(0)
#define FCASE(name,idx,exp,got) do{float e_=(float)(exp),g_=(float)(got);if(e_!=g_){fprintf(stderr,"case %s index %zu expected %.9g got %.9g\n",name,(size_t)(idx),(double)e_,(double)g_);return 1;}}while(0)
typedef struct {
    float           v;
    int             i;
} Item;
static int      cmp(const void *a, const void *b){
    const           Item *x = a, *y = b;
    if (x->v > y->v)
        return -1;
    if (x->v < y->v)
        return 1;
    return (x->i > y->i) - (x->i < y->i);
}
static int      verify(const char *name, const float *v, int n, int k){
    int            *ix = malloc((size_t) k * sizeof *ix);
    float          *rank = malloc((size_t) k * sizeof *rank);
    Item           *items = malloc((size_t) n * sizeof *items);
    if (!ix || !rank || !items)
        return 2;
    for (int i = 0; i < n; i++) {
        items[i].v = v[i];
        items[i].i = i;
    } qsort(items, (size_t) n, sizeof *items, cmp);
    int             rc = rank_largest_topk(v, n, k, ix, rank);
    ICASE(name, 0, 0, rc);
    for (int j = 0; j < k; j++) {
        ICASE(name, (size_t) j, items[j].i, ix[j]);
        FCASE(name, (size_t) j, items[j].v, rank[j]);
    } free(ix);
    free(rank);
    free(items);
    return 0;
}
int
main(void)
{
    {
        float           v[] = {4, 9, 9, -1, 3};
        if (verify("stable-tied-top-k", v, 5, 3))
            return 1;
        if (verify("k-equals-n", v, 5, 5))
            return 1;
        if (verify("k-one", v, 5, 1))
            return 1;
    }
    {
        float           v[] = {1, 2, 3};
        int             ix[3] = {51, 52, 53};
        float           rank[3] = {61, 62, 63};
        ICASE("null-values", 0, -1, rank_largest_topk(NULL, 3, 2, ix, rank));
        ICASE("null-indices", 0, -1, rank_largest_topk(v, 3, 2, NULL, rank));
        ICASE("null-ranked", 0, -1, rank_largest_topk(v, 3, 2, ix, NULL));
        ICASE("null-outputs-atomic", 0, 51, ix[0]);
        ICASE("null-outputs-atomic", 1, 61, rank[0]);
        ICASE("zero-n", 0, -1, rank_largest_topk(v, 0, 1, ix, rank));
        ICASE("negative-n", 0, -1, rank_largest_topk(v, -1, 1, ix, rank));
        ICASE("zero-k", 0, -1, rank_largest_topk(v, 3, 0, ix, rank));
        ICASE("negative-k", 0, -1, rank_largest_topk(v, 3, -1, ix, rank));
        ICASE("k-too-large", 0, -1, rank_largest_topk(v, 3, 4, ix, rank));
        ICASE("invalid-k-atomic", 0, 51, ix[0]);
        ICASE("invalid-k-atomic", 1, 61, rank[0]);
        float           bad[] = {1, 2, NAN};
        ICASE("nan-input", 0, -1, rank_largest_topk(bad, 3, 1, ix, rank));
        ICASE("nan-input-atomic", 0, 51, ix[0]);
        ICASE("nan-input-atomic", 1, 61, rank[0]);
        float           inf[] = {1, INFINITY, 2};
        ICASE("infinite-input", 0, -1, rank_largest_topk(inf, 3, 1, ix, rank));
    }
    {
        float           v[4] = {9, 8, 7, 6}, rank[4] = {-1, -2, -3, -4};
        int             ix[4] = {91, 92, 93, 94};
        ICASE("writes-exactly-k", 0, 0, rank_largest_topk(v, 4, 2, ix, rank));
        ICASE("writes-exactly-k-index-0", 0, 0, ix[0]);
        ICASE("writes-exactly-k-index-1", 1, 1, ix[1]);
        ICASE("writes-exactly-k-guard-index", 2, 93, ix[2]);
        ICASE("writes-exactly-k-guard-value", 2, -3, rank[2]);
    }
    for (size_t c = 0; c < 240; c++) {
        int             n = 1 + (int)(rnd() % 300), k = 1 + (int)(rnd() % (unsigned)n);
        float          *v = malloc((size_t) n * sizeof *v);
        if (!v)
            return 2;
        for (int i = 0; i < n; i++)
            v[i] = (float)((int)(rnd() % 41) - 20);
        if (verify("random-tied-ranking", v, n, k))
            return 1;
        free(v);
    }
    {
        int             n = 65537, k = 257;
        float          *v = malloc((size_t) n * sizeof *v);
        if (!v)
            return 2;
        for (int i = 0; i < n; i++)
            v[i] = (float)((int)(rnd() % 2001) - 1000);
        v[n - 1] = 1.0e20f;
        if (verify("large-ranking", v, n, k))
            return 1;
        free(v);
    }
    if (tracked_heap_allocations == SIZE_MAX) {
        heldout_test_malloc(1);
        heldout_test_calloc(1, 1);
        heldout_test_realloc(NULL, 1);
    }
    ICASE("no-heap-allocation", 0, 0, tracked_heap_allocations);
    return 0;
}
