#include <stddef.h>
typedef struct { int *data; size_t capacity; size_t size; size_t next; } TraceRing;
int trail_ring_init(TraceRing *r, int *storage, size_t capacity) {
    if (r == 0 || storage == 0 || capacity == 0) return -1;
    r->data=storage; r->capacity=capacity; r->size=0; r->next=0; return 0;
}
int trail_ring_push(TraceRing *r, int value) {
    if (r == 0 || r->data == 0 || r->capacity == 0 || r->next >= r->capacity || r->size > r->capacity) return -1;
    r->data[r->next]=value; r->next=(r->next+1)%r->capacity;
    if (r->size < r->capacity) ++r->size;
    return 0;
}
int trail_ring_get(const TraceRing *r, size_t age, int *out) {
    size_t index;
    if (r == 0 || r->data == 0 || out == 0 || r->capacity == 0 || r->next >= r->capacity || r->size > r->capacity || age >= r->size) return -1;
    index=(r->next + r->capacity - 1 - age)%r->capacity; *out=r->data[index]; return 0;
}
