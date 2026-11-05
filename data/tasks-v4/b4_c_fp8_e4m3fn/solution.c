#include <math.h>
#include <stdint.h>
uint8_t float_to_e4m3fn(float x) {
    uint8_t sign = signbit(x) ? 0x80u : 0u;
    double value = fabs((double)x), best_diff = 1.0e300;
    unsigned best = 0;
    for (unsigned code = 0; code <= 0x7eu; ++code) {
        unsigned exponent = code >> 3, mantissa = code & 7u;
        double candidate = exponent == 0 ? (double)mantissa / 512.0
                         : ldexp(1.0 + (double)mantissa / 8.0, (int)exponent - 7);
        double diff = fabs(value - candidate);
        if (diff < best_diff || (diff == best_diff && (code & 1u) == 0u && (best & 1u) != 0u)) {
            best = code; best_diff = diff;
        }
    }
    return (uint8_t)(sign | best);
}
