#include "qengine.h"
#include <stdlib.h>
#include <sys/resource.h>
#include "expert_store.h"
#include "model.h"
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct { Model model; int pos; } Handle;   /* Model is the first member: QeModel* == Handle* */

QeModel *qe_load(const char *dir, int max_len) {
    Handle *h = calloc(1, sizeof *h);
    if (!h || max_len < 1) { free(h); return NULL; }
    if (model_load(&h->model, dir, max_len) != 0) { free(h); return NULL; }   /* load failures leak by design: process exits */
    return (QeModel *)h;
}

void qe_free(QeModel *m) { free(m); }   /* weights are released at process exit */

void qe_reset(QeModel *m) { model_reset(m); ((Handle *)m)->pos = 0; }

int qe_forward(QeModel *m, int token, float *logits) {
    Handle *h = (Handle *)m;
    if (h->pos >= h->model.max_len) return -2;
    if (model_forward(&h->model, token, logits) != 0) return -1;
    h->pos++;
    return 0;
}

int qe_vocab(const QeModel *m) { return m->cfg.vocab; }
int qe_position(const QeModel *m) { return ((const Handle *)m)->pos; }
int qe_max_len(const QeModel *m) { return m->max_len; }

int qe_set_threads(int n) {
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
    return omp_get_max_threads();
#else
    (void)n;
    return 1;
#endif
}

double qe_expert_gb_read(void) { return expert_bytes_read() / 1e9; }
double qe_expert_read_seconds(void) { return expert_read_seconds(); }
double qe_peak_rss_gb(void) { struct rusage ru; getrusage(RUSAGE_SELF, &ru); return ru.ru_maxrss / 1073741824.0; }
