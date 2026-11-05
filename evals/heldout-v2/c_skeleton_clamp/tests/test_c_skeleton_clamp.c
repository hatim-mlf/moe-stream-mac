#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "../candidate.c"
static uint32_t state = 0x1940b761u;
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
static int      verify(const char *name, const float *in, size_t n, float lo, float hi, float *out, size_t * count){
    size_t          ecount = 0;
    for (size_t i = 0; i < n; i++) {
        if (in[i] < lo || in[i] > hi)
            ecount++;
    } ICASE(name, 0, 0, clamp_and_count(in, n, lo, hi, out, count));
    ICASE(name, 1, ecount, *count);
    for (size_t i = 0; i < n; i++) {
        float           e = in[i] < lo ? lo : in[i] > hi ? hi : in[i];
        if (fcase(name, i, e, out[i]))
            return 1;
    } return 0;
}
int
main(void)
{
    if (!fixed_lines_ok()) {
        fprintf(stderr, "case skeleton-fixed-lines index 0 expected template non-TODO bytes identical got mismatch\n");
        return 1;
    } {
        float           in[] = {-2, 0.5f, 4, 1}, out[4] = {9, 9, 9, 9};
        size_t          count = 99;
        if (verify("clamp-strict-count", in, 4, 0, 2, out, &count))
            return 1;
        float           edge[] = {0, 2};
        if (verify("exact-boundaries-not-counted", edge, 2, 0, 2, out, &count))
            return 1;
    }
    {
        float           in[2] = {-1, 1}, out[2] = {71, 72};
        size_t          count = 73;
        ICASE("null-input", 0, -1, clamp_and_count(NULL, 2, 0, 1, out, &count));
        ICASE("null-output", 0, -1, clamp_and_count(in, 2, 0, 1, NULL, &count));
        ICASE("null-count", 0, -1, clamp_and_count(in, 2, 0, 1, out, NULL));
        ICASE("null-errors-atomic", 0, 71, out[0]);
        ICASE("null-errors-atomic", 1, 72, out[1]);
        ICASE("null-errors-count-atomic", 0, 73, count);
        ICASE("zero-length", 0, -1, clamp_and_count(in, 0, 0, 1, out, &count));
        ICASE("zero-length-atomic", 0, 71, out[0]);
        ICASE("reversed-bounds", 0, -1, clamp_and_count(in, 2, 2, 1, out, &count));
        ICASE("nan-lower", 0, -1, clamp_and_count(in, 2, NAN, 1, out, &count));
        ICASE("infinite-upper", 0, -1, clamp_and_count(in, 2, 0, INFINITY, out, &count));
        ICASE("invalid-bounds-atomic", 0, 71, out[0]);
        ICASE("invalid-bounds-count-atomic", 0, 73, count);
        float           bad[] = {0, NAN};
        ICASE("nonfinite-input", 0, -1, clamp_and_count(bad, 2, 0, 1, out, &count));
        ICASE("nonfinite-input-atomic", 0, 71, out[0]);
        ICASE("nonfinite-count-atomic", 0, 73, count);
        float           inf[] = {INFINITY, 0};
        ICASE("infinite-input", 0, -1, clamp_and_count(inf, 2, 0, 1, out, &count));
    }
    for (size_t c = 0; c < 240; c++) {
        size_t          n = 1 + rnd() % 257;
        float          *in = malloc(n * sizeof *in), *out = malloc(n * sizeof *out);
        if (!in || !out)
            return 2;
        float           lo = ((int)(rnd() % 2001) - 1000) / 100.0f, hi = lo + (rnd() % 1000) / 100.0f;
        for (size_t i = 0; i < n; i++)
            in[i] = ((int)(rnd() % 6001) - 3000) / 100.0f;
        if (verify("random-clamp-property", in, n, lo, hi, out, &n))
            return 1;
        free(in);
        free(out);
    }
    {
        size_t          n = 65537;
        float          *in = malloc(n * sizeof *in), *out = malloc(n * sizeof *out);
        if (!in || !out)
            return 2;
        for (size_t i = 0; i < n; i++)
            in[i] = (float)((int)(i % 2001) - 1000) / 7.0f;
        size_t          count = 0;
        if (verify("largest-clamp-vector", in, n, -50, 50, out, &count))
            return 1;
        free(in);
        free(out);
    }
    return 0;
}
