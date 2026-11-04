/* run_model: greedy generation with the engine, saving logits at every position.
 *   run_model <model dir> <ids.i32> <n_gen> <out.bin>
 * out: int32 n_prompt, n_gen, vocab; int32 ids[n]; float32 logits[n-1][vocab]  (row t = after ids[0..t]) */
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <time.h>
#include "expert_store.h"
#include "model.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int argmax(const float *v, int n) { int b = 0; for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b; }

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s model ids.i32 n_gen out.bin\n", argv[0]); return 2; }
    FILE *f = fopen(argv[2], "rb");
    if (!f) { perror(argv[2]); return 1; }
    int32_t ids[4096]; const int np = (int)fread(ids, 4, 2048, f); fclose(f);
    const int ng = atoi(argv[3]), n = np + ng;
    double t0 = now(); Model m;
    if (np < 1 || model_load(&m, argv[1], n + 8)) { fprintf(stderr, "load failed\n"); return 1; }
    const int V = m.cfg.vocab;
    fprintf(stderr, "loaded in %.1fs\n", now() - t0);
    float *logits = malloc(sizeof(float) * (size_t)(n - 1) * V);
    double t_prompt = now(), t_gen = 0;
    for (int t = 0; t < n - 1; t++) {
        if (t == np) t_gen = now();
        if (model_forward(&m, ids[t], logits + (size_t)t * V)) { fprintf(stderr, "forward failed at %d\n", t); return 1; }
        if (t >= np - 1) ids[t + 1] = argmax(logits + (size_t)t * V, V);
    }
    const double end = now();
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    fprintf(stderr, "prompt %d tok: %.2f s/tok | generate %d tok: %.2f s/tok | experts read %.1f GB in %.1f s (%.0f%% of run) | peak RSS %.2f GB\n",
            np, (t_gen - t_prompt) / np, ng - 1, (end - t_gen) / (ng - 1 > 0 ? ng - 1 : 1),
            expert_bytes_read() / 1e9, expert_read_seconds(), 100.0 * expert_read_seconds() / (end - t_prompt), ru.ru_maxrss / 1073741824.0);
    FILE *o = fopen(argv[4], "wb");
    const int32_t hdr[3] = { np, ng, V };
    if (!o || fwrite(hdr, 4, 3, o) != 3 || fwrite(ids, 4, n, o) != (size_t)n ||
        fwrite(logits, sizeof(float), (size_t)(n - 1) * V, o) != (size_t)(n - 1) * V) return 1;
    fclose(o);
    return 0;
}
