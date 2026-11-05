#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    uint8_t storage[4] = {0}, out[4] = {0}, a[3] = {1,2,3}, b[3] = {4,5,6};
    ByteRing q; byte_ring_init(&q, storage, 4);
    assert(byte_ring_write(&q, a, 3) == 0);
    assert(byte_ring_read(&q, out, 2) == 0 && out[0] == 1 && out[1] == 2);
    assert(byte_ring_write(&q, b, 3) == 0 && q.size == 4);
    assert(byte_ring_write(&q, a, 1) == -1 && q.size == 4);
    assert(byte_ring_read(&q, out, 4) == 0);
    assert(out[0] == 3 && out[1] == 4 && out[2] == 5 && out[3] == 6 && q.size == 0);
    assert(byte_ring_read(&q, out, 1) == -1 && q.size == 0);
    assert(byte_ring_write(&q, NULL, 0) == 0 && byte_ring_read(&q, NULL, 0) == 0);
    ByteRing invalid; byte_ring_init(&invalid, storage, 0);
    assert(byte_ring_write(&invalid, NULL, 0) == -1);
    return 0;
}
