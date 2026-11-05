Target: candidate.c (C99)

Implement a caller-buffer-backed ring that reads values by age.

Exact signature:
```c
typedef struct { int *data; size_t capacity; size_t size; size_t next; } TraceRing;
int trail_ring_init(TraceRing *r, int *storage, size_t capacity);
int trail_ring_push(TraceRing *r, int value);
int trail_ring_get(const TraceRing *r, size_t age, int *out);
```

`init` requires non-null ring/storage and positive capacity, then sets an empty ring. `push` writes one value, overwriting the oldest when full. `get` uses age 0 for newest and age `size-1` for oldest; it returns -1 for an invalid ring, null output, or unavailable age and leaves output unchanged on error. Each operation returns 0 on success and -1 on invalid input. Use the supplied storage only; do not allocate.

Return only the complete file in one fenced c block.
