/* test_qgroup: qgroup_dot vs a double-precision scalar reference, random and edge inputs. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "qgroup.h"

static double ref(const uint32_t *w, const float *x, int n, int bits) {
    const int per = 32 / bits; const uint32_t mask = (1u << bits) - 1u; double s = 0;
    for (int j = 0; j < n; j++) s += (double)((w[j / per] >> ((j % per) * bits)) & mask) * x[j];
    return s;
}

int main(void) {
    uint32_t w[32]; float x[128]; int failed = 0, cases = 0;
    srand(42);
    const int ns[3] = { 32, 64, 128 }, bitss[2] = { 4, 8 };
    for (int trial = 0; trial < 300; trial++)
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 2; b++) {
                const int n = ns[a], bits = bitss[b];
                for (int i = 0; i < 32; i++) {
                    const uint32_t r = ((uint32_t)rand() << 16) ^ (uint32_t)rand();
                    w[i] = trial == 0 ? 0u : trial == 1 ? 0xFFFFFFFFu : r;   /* edges: all 0, all max */
                }
                for (int i = 0; i < n; i++) x[i] = (float)((double)rand() / RAND_MAX * 4.0 - 2.0);
                const double want = ref(w, x, n, bits); const float got = qgroup_dot(w, x, n, bits);
                const double tol = 1e-5 * (fabs(want) + 1.0) * n;
                cases++;
                if (!(fabs(got - want) <= tol)) {
                    if (failed++ < 5) printf("FAIL  trial %d n=%d bits=%d: got %.6f want %.6f\n", trial, n, bits, got, want);
                }
            }
    printf(failed ? "QGROUP TESTS FAILED (%d/%d)\n" : "ALL QGROUP TESTS PASSED (%d/%d cases)\n", failed ? failed : cases, cases);
    return failed != 0;
}
