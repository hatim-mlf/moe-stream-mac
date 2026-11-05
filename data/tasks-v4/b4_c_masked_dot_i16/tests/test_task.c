#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    const int16_t a[4] = {-300, 2, 100, 7}, b[4] = {4, 9, -5, 8};
    const uint8_t m[4] = {1, 0, 3, 0};
    assert(masked_dot_i16(a, b, m, 4) == -1700);
    assert(masked_dot_i16(NULL, NULL, NULL, 0) == 0);
    return 0;
}
