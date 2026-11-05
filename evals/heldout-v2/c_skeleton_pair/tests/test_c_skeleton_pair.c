#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include "../candidate.c"
static uint32_t state = 0xa412df29u;
static uint32_t rnd(void){
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
#define ICASE(name,idx,exp,got) do{long long e_=(long long)(exp),g_=(long long)(got);if(e_!=g_){fprintf(stderr,"case %s index %zu expected %lld got %lld\n",name,(size_t)(idx),e_,g_);return 1;}}while(0)
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
static long     naive_int(const char **p){
    int             neg = 0;
    long            v = 0;
    if (**p == '+' || **p == '-') {
        neg = **p == '-';
        (*p)++;
    } while (**p >= '0' && **p <= '9') {
        v = v * 10 + (**p - '0');
        (*p)++;
    } return neg ? -v : v;
}
static int      pair(const char *s, int16_t * left, int16_t * right){
    const char     *p = s;
    long            a = naive_int(&p);
    if (*p++ != ',')
        return 1;
    long            b = naive_int(&p);
    if (*p)
        return 1;
    *left = (int16_t) a;
    *right = (int16_t) b;
    return 0;
}
static int      success_case(const char *name, const char *s, int e1, int e2){
    int16_t         a = 111, b = 222;
    ICASE(name, 0, 0, parse_pair_record(s, &a, &b));
    ICASE(name, 1, e1, a);
    ICASE(name, 2, e2, b);
    return 0;
}
static int      error_case(const char *name, const char *s, int rc){
    int16_t         a = 111, b = 222;
    ICASE(name, 0, rc, parse_pair_record(s, &a, &b));
    ICASE(name, 1, 111, a);
    ICASE(name, 2, 222, b);
    return 0;
}
int             main(void){
    if (!fixed_lines_ok()) {
        fprintf(stderr, "case skeleton-fixed-lines index 0 expected template non-TODO bytes identical got mismatch\n");
        return 1;
    }
    if (success_case("signed-min-max", "-32768,+32767", -32768, 32767) || success_case("unsigned-decimal", "0,12", 0, 12) || success_case("both-signs", "+1,-2", 1, -2) || success_case("leading-zeroes", "-0002,+0003", -2, 3))
        return 1;
    if (error_case("null-text", NULL, -1)) {
        return 1;
    }
    {
        int16_t         a = 11, b = 22;
        ICASE("null-left", 0, -1, parse_pair_record("1,2", NULL, &b));
        ICASE("null-right", 0, -1, parse_pair_record("1,2", &a, NULL));
        ICASE("null-output-atomic", 0, 11, a);
        ICASE("null-output-atomic", 1, 22, b);
        }
        if (error_case("empty-text", "", -2) || error_case("missing-first-digit", ",2", -2) || error_case("missing-comma", "1 2", -2) || error_case("missing-second-digit", "1,", -2) || error_case("extra-comma", "1,2,3", -2) || error_case("trailing-text", "1,2x", -2) || error_case("leading-whitespace", " 1,2", -2) || error_case("embedded-whitespace", "1, 2", -2) || error_case("double-sign", "--1,2", -2) || error_case("sign-without-digit", "+,2", -2) || error_case("positive-overflow-left", "32768,2", -3) || error_case("negative-overflow-left", "-32769,2", -3) || error_case("positive-overflow-right", "1,32768", -3) || error_case("negative-overflow-right", "1,-32769", -3))
            return 1;
    for (size_t c = 0; c < 256; c++) {
        int             a = (int)(rnd() % 65536) - 32768, b = (int)(rnd() % 65536) - 32768;
        char            s[64];
        snprintf(s, sizeof s, "%+d,%+d", a, b);
        int16_t         ea = 0, eb = 0, ga = 0, gb = 0;
        pair(s, &ea, &eb);
        ICASE("random-valid-parser", c, 0, parse_pair_record(s, &ga, &gb));
        ICASE("random-valid-parser-left", c, ea, ga);
        ICASE("random-valid-parser-right", c, eb, gb);
    }
    return 0;
}
