#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    assert(multiply_q15_round(16384,16384)==8192);
    assert(multiply_q15_round(1,16384)==1);
    assert(multiply_q15_round(-1,16384)==-1);
    assert(multiply_q15_round(INT16_MIN,INT16_MIN)==INT16_MAX);
}
