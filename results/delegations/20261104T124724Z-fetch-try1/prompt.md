# Task: write ONE C function, fetch_experts, for engine/src/moe_layer.c

Context (already in the file; do not repeat it):

```c
#define MOE_MAX_TOPK 16
typedef struct { uint32_t *w; uint16_t *s, *b; } ExpertBuf;
typedef struct { /* ... */ } ExpertTensor;
typedef struct {
    /* ... */
    ExpertTensor gate, up, down;                      /* routed experts on disk */
    ExpertBuf gbuf[MOE_MAX_TOPK], ubuf[MOE_MAX_TOPK], dbuf[MOE_MAX_TOPK];  /* one set per selected expert */
    int hidden, E, top_k, inter, sh_inter;
} MoeWeights;

/* Thread-safe: reads expert e of t into buf with pread and returns a QMat view in *out. 0 or -1. */
int expert_tensor_read(const ExpertTensor *t, int e, const ExpertBuf *buf, QMat *out);
```

Write exactly this function:

```c
/* Read the gate, up and down matrices of the top_k selected experts idx[0..top_k) into their own buffers
 * (w->gbuf[i], w->ubuf[i], w->dbuf[i]) IN PARALLEL, and return QMat views in g[i], u[i], d[i].
 * Returns 0, or -1 if any read failed. */
static int fetch_experts(const MoeWeights *w, const int *idx, QMat *g, QMat *u, QMat *d)
```

Requirements:
- Parallelise with OpenMP over the 3 * top_k reads: `#pragma omp parallel for schedule(dynamic, 1) reduction(|:failed)`
  over j in [0, 3*top_k): i = j / 3, which = j % 3 (0 gate, 1 up, 2 down).
- Each read uses ONLY slot i's buffer for that matrix, so no two threads share a buffer.
- `int failed = 0;` accumulate `failed |= (expert_tensor_read(...) != 0);` and return failed ? -1 : 0.
- No allocation, no globals, under 25 lines.

Return only this function in one fenced c block.
