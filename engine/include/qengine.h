/* qengine.h: stable C API of the engine, for front ends (the Python terminal chat uses it via ctypes). */
#ifndef QENGINE_H
#define QENGINE_H
#include <stddef.h>

typedef struct Model QeModel;

/* Load a qwen3_5_moe MLX checkpoint. max_len = context length (tokens). NULL on failure (stderr says why). */
QeModel *qe_load(const char *dir, int max_len);
void qe_free(QeModel *m);
/* Forget the conversation: clears KV caches and recurrent state. */
void qe_reset(QeModel *m);
/* Feed one token; logits receives qe_vocab(m) floats. 0 ok, -1 error, -2 context full (call qe_reset). */
int qe_forward(QeModel *m, int token, float *logits);
int qe_vocab(const QeModel *m);
int qe_position(const QeModel *m);   /* tokens fed since the last reset */
int qe_max_len(const QeModel *m);
/* Threads used by the math kernels from now on (OpenMP). Returns the effective count. */
int qe_set_threads(int n);
/* I/O statistics since process start. */
double qe_expert_gb_read(void);
double qe_expert_read_seconds(void);
double qe_peak_rss_gb(void);

#endif
