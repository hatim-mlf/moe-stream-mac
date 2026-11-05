#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "../candidate.c"
static uint32_t state = 0x114a809bu;
static uint32_t rnd(void){
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
#define ICASE(name,idx,exp,got) do{long long e_=(long long)(exp),g_=(long long)(got);if(e_!=g_){fprintf(stderr,"case %s index %zu expected %lld got %lld\n",name,(size_t)(idx),e_,g_);return 1;}}while(0)
/* Float comparisons use relative tolerance 1e-5 plus absolute tolerance 1e-6. */
static int      fcase(const char *n, size_t i, float e, float g){
    float           tol = 1e-6f + 1e-5f * fabsf(e);
    if (!(fabsf(e - g) <= tol)) {
        fprintf(stderr, "case %s index %zu expected %.9g got %.9g\n", n, i, (double)e, (double)g);
        return 1;
    } return 0;
}
static int fixed_lines_ok(void)
{
    FILE *candidate = fopen("candidate.c", "r");
    FILE *template = fopen("tests/template.c", "r");
    char candidate_line[512], template_line[512];
    int ok = candidate != NULL && template != NULL;

    if (!ok) {
        if (candidate != NULL) fclose(candidate);
        if (template != NULL) fclose(template);
        return 0;
    }
    while (fgets(template_line, sizeof template_line, template) != NULL) {
        if (fgets(candidate_line, sizeof candidate_line, candidate) == NULL ||
            strcmp(candidate_line, template_line) != 0) {
            ok = 0;
            break;
        }
        if (strstr(template_line, "TODO_BEGIN") != NULL) {
            int found_end = 0;
            while (fgets(candidate_line, sizeof candidate_line, candidate) != NULL) {
                if (strstr(candidate_line, "TODO_END") != NULL) {
                    found_end = 1;
                    break;
                }
            }
            if (!found_end || fgets(template_line, sizeof template_line, template) == NULL ||
                strstr(template_line, "TODO_END") == NULL ||
                strcmp(candidate_line, template_line) != 0) {
                ok = 0;
                break;
            }
        }
    }
    if (ok && fgets(candidate_line, sizeof candidate_line, candidate) != NULL) ok = 0;
    fclose(candidate);
    fclose(template);
    return ok;
}
static int      verify(const char *name, const float *x, size_t n, size_t w, float *out){
    size_t          count = n - w + 1;
    ICASE(name, 0, 0, window_energy(x, n, w, out, count));
    for (size_t i = 0; i < count; i++) {
        double          sum = 0;
        for (size_t j = 0; j < w; j++)
            sum += (double)x[i + j] * x[i + j];
        if (fcase(name, i, (float)(sum / w), out[i]))
            return 1;
    } return 0;
}
int             main(void){
    if (!fixed_lines_ok()) {
        fprintf(stderr, "case skeleton-fixed-lines index 0 expected template non-TODO bytes identical got mismatch\n");
        return 1;
    } {
        float           x[] = {1, 2, 3, 4}, out[4] = {71, 72, 73, 74};
        if (verify("adjacent-windows", x, 4, 2, out) || verify("width-one", x, 4, 1, out) || verify("width-equals-n", x, 4, 4, out))
            return 1;
    }
    {
        float           x[4] = {1, 2, 3, 4}, out[4] = {51, 52, 53, 54};
        ICASE("null-input", 0, -1, window_energy(NULL, 4, 2, out, 3));
        ICASE("null-output", 0, -1, window_energy(x, 4, 2, NULL, 3));
        ICASE("zero-n", 0, -1, window_energy(x, 0, 1, out, 0));
        ICASE("zero-width", 0, -1, window_energy(x, 4, 0, out, 5));
        ICASE("width-too-large", 0, -1, window_energy(x, 4, 5, out, 0));
        ICASE("short-output", 0, -1, window_energy(x, 4, 2, out, 2));
        ICASE("long-output", 0, -1, window_energy(x, 4, 2, out, 4));
        ICASE("invalid-atomic", 0, 51, out[0]);
        ICASE("invalid-atomic", 3, 54, out[3]);
        float           bad[] = {1, 2, INFINITY, 4};
        ICASE("nonfinite-input", 0, -1, window_energy(bad, 4, 2, out, 3));
        float           nanx[] = {1, NAN, 3, 4};
        ICASE("nan-input", 0, -1, window_energy(nanx, 4, 2, out, 3));
        ICASE("nonfinite-atomic", 0, 51, out[0]);
        ICASE("nonfinite-atomic", 2, 53, out[2]);
    }
    for (size_t c = 0; c < 240; c++) {
        size_t          n = 1 + rnd() % 129, w = 1 + rnd() % n, count = n - w + 1;
        float          *x = malloc(n * sizeof *x), *out = malloc(count * sizeof *out);
        if (!x || !out)
            return 2;
        for (size_t i = 0; i < n; i++)
            x[i] = ((int)(rnd() % 4001) - 2000) / 64.0f;
        if (verify("random-window-energy", x, n, w, out))
            return 1;
        free(x);
        free(out);
    }
    {
        size_t          n = 65537, w = 257, count = n - w + 1;
        float          *x = malloc(n * sizeof *x), *out = malloc(count * sizeof *out);
        if (!x || !out)
            return 2;
        for (size_t i = 0; i < n; i++)
            x[i] = (float)((int)(i % 101) - 50) / 13.0f;
        if (verify("largest-window-vector", x, n, w, out))
            return 1;
        free(x);
        free(out);
    }
    return 0;
}
