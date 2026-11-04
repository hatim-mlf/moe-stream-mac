```c
static int fetch_experts(const MoeWeights *w, const int *idx, QMat *g, QMat *u, QMat *d) {
    int failed = 0;
    #pragma omp parallel for schedule(dynamic, 1) reduction(|:failed)
    for (int j = 0; j < 3 * w->top_k; ++j) {
        int i = j / 3;
        int which = j % 3;
        int expert = idx[i];
        ExpertBuf *buf = (which == 0) ? &w->gbuf[i] : (which == 1) ? &w->ubuf[i] : &w->dbuf[i];
        const ExpertTensor *t = (which == 0) ? &w->gate : (which == 1) ? &w->up : &w->down;
        QMat *out = (which == 0) ? &g[i] : (which == 1) ? &u[i] : &d[i];
        failed |= (expert_tensor_read(t, expert, buf, out) != 0);
    }
    return failed ? -1 : 0;
}
```