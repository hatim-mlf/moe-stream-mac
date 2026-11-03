/* test_model: whole-model forward vs tests/model_ref.py (MLX, float32).
 *   test_model <model dir> <ref.bin>
 * Checks (1) logits at every position (relative L2 + same argmax), (2) the engine's own greedy
 * continuation from the prompt reproduces MLX's generated tokens exactly. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "model.h"

#define REL_TOL 5e-2f   /* MLX reference runs bf16 activations; argmax and greedy tokens must still match exactly */

static int argmax(const float *v, int n) { int b = 0; for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b; }

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s model ref.bin\n", argv[0]); return 2; }
    FILE *f = fopen(argv[2], "rb"); int32_t hdr[3];
    if (!f || fread(hdr, 4, 3, f) != 3) { fprintf(stderr, "bad ref\n"); return 1; }
    const int np = hdr[0], ng = hdr[1], V = hdr[2], n = np + ng;
    int32_t *ids = malloc(4 * (size_t)n); float *want = malloc(4 * (size_t)(n - 1) * V);
    if (fread(ids, 4, n, f) != (size_t)n || fread(want, 4, (size_t)(n - 1) * V, f) != (size_t)(n - 1) * V) return 1;
    fclose(f);
    double t0 = now(); Model m;
    if (model_load(&m, argv[1], 512)) { fprintf(stderr, "model_load failed\n"); return 1; }
    if (m.cfg.vocab != V) { fprintf(stderr, "vocab mismatch %d vs %d\n", m.cfg.vocab, V); return 1; }
    printf("loaded %d layers in %.1fs\n", m.cfg.n_layers, now() - t0);
    float *logits = malloc(4 * (size_t)V); int failed = 0;
    t0 = now();
    for (int t = 0; t < n - 1; t++) {                       /* (1) teacher forcing */
        if (model_forward(&m, ids[t], logits)) { printf("FAIL forward at %d\n", t); return 1; }
        const float *w = want + (size_t)t * V; double num = 0, den = 0;
        for (int i = 0; i < V; i++) { double e = logits[i] - w[i]; num += e * e; den += (double)w[i] * w[i]; }
        const float rel = (float)sqrt(num / den); const int same = argmax(logits, V) == argmax(w, V);
        failed |= !(rel < REL_TOL) || !same;
        printf("%s  pos %2d: rel L2 %.2e, argmax %s\n", rel < REL_TOL && same ? "PASS" : "FAIL", t, rel, same ? "same" : "DIFFERENT");
    }
    printf("teacher forcing: %.2f s/token\n", (now() - t0) / (n - 1));
    model_reset(&m);                                        /* (2) engine's own greedy continuation */
    int tok = 0;
    for (int t = 0; t < np; t++) model_forward(&m, ids[t], logits);
    int match = 0;
    for (int g = 0; g < ng; g++) {
        tok = argmax(logits, V);
        match += tok == ids[np + g];
        printf("  gen %2d: engine %6d  mlx %6d %s\n", g, tok, ids[np + g], tok == ids[np + g] ? "" : "<-- differs");
        if (model_forward(&m, tok, logits)) return 1;
    }
    failed |= match != ng;
    printf("%s  greedy continuation: %d/%d tokens identical to MLX\n", match == ng ? "PASS" : "FAIL", match, ng);
    printf(failed ? "MODEL TEST FAILED\n" : "WHOLE MODEL MATCHES MLX\n");
    return failed;
}
