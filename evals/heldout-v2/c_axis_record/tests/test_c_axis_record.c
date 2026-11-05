#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "../candidate.c"
static unsigned rng = 0x24c0ffeeu;
static unsigned
rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}
#define ICASE(name,idx,exp,got) do { long long e_=(long long)(exp),g_=(long long)(got); if(e_!=g_){fprintf(stderr,"case %s index %zu expected %lld got %lld\n",name,(size_t)(idx),e_,g_);return 1;} } while(0)
static int      parse_case(const char *name, const char *s, int rc, int ea, int eb, int ec){
    int             a = 111, b = 222, c = 333;
    int             got = parse_axis_record(s, &a, &b, &c);
    ICASE(name, 0, rc, got);
    if (rc == 0) {
        ICASE(name, 1, ea, a);
        ICASE(name, 2, eb, b);
        ICASE(name, 3, ec, c);
    } else {
        ICASE(name, 1, 111, a);
        ICASE(name, 2, 222, b);
        ICASE(name, 3, 333, c);
    } return 0;
}
int
main(void)
{
    if (parse_case("valid-basic", "A3,12,20", 0, 3, 12, 20))
        return 1;
    if (parse_case("valid-leading-zeroes", "A07,000,001", 0, 7, 0, 1))
        return 1;
    if (parse_case("valid-boundaries", "A7,2147483646,2147483647", 0, 7, 2147483646, 2147483647))
        return 1;
    if (parse_case("invalid-prefix", "B1,2,3", -2, 0, 0, 0) || parse_case("invalid-empty-axis", "A,1,2", -2, 0, 0, 0) || parse_case("invalid-negative", "A-1,0,1", -2, 0, 0, 0) || parse_case("invalid-space", "A1, 2,3", -2, 0, 0, 0) || parse_case("invalid-missing-comma", "A1,2 3", -2, 0, 0, 0) || parse_case("invalid-trailing", "A1,2,3x", -2, 0, 0, 0) || parse_case("invalid-empty-range", "A1,4,4", -3, 0, 0, 0) || parse_case("invalid-reversed-range", "A1,5,4", -3, 0, 0, 0) || parse_case("invalid-axis-range", "A8,0,1", -3, 0, 0, 0) || parse_case("invalid-int-overflow-start", "A1,2147483648,2147483649", -3, 0, 0, 0) || parse_case("invalid-int-overflow-end", "A1,0,999999999999999999999999", -3, 0, 0, 0))
        return 1;
    {
        int             a = 1, b = 2, c = 3;
        ICASE("null-text", 0, -1, parse_axis_record(NULL, &a, &b, &c));
        ICASE("null-text", 1, 1, a);
        ICASE("null-text", 2, 2, b);
        ICASE("null-text", 3, 3, c);
        ICASE("null-axis", 0, -1, parse_axis_record("A1,0,1", NULL, &b, &c));
        ICASE("null-start", 0, -1, parse_axis_record("A1,0,1", &a, NULL, &c));
        ICASE("null-end", 0, -1, parse_axis_record("A1,0,1", &a, &b, NULL));
        ICASE("null-outs", 1, 1, a);
        ICASE("null-outs", 2, 2, b);
        ICASE("null-outs", 3, 3, c);
    }
    for (size_t k = 0; k < 256; k++) {
        int             axis = (int)(rnd() % 8), start = (int)(rnd() % 100000), end = start + 1 + (int)(rnd() % 100000);
        char            s[128];
        int             a = -1, b = -1, c = -1;
        snprintf(s, sizeof s, "A%d,%d,%d", axis, start, end);
        int             rc = parse_axis_record(s, &a, &b, &c);
        ICASE("random-valid-record", k, 0, rc);
        ICASE("random-valid-record", 3 * k, axis, a);
        ICASE("random-valid-record", 3 * k + 1, start, b);
        ICASE("random-valid-record", 3 * k + 2, end, c);
    }
    return 0;
}
