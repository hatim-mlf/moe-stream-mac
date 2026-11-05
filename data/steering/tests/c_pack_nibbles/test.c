#include <assert.h>
#include <stddef.h>
#include <stdint.h>
int pack_nibbles(uint8_t, uint8_t, uint8_t *);
int main(void) { uint8_t out = 0; assert(pack_nibbles(0x0a, 0x03, &out) == 0 && out == 0xa3); out = 0x55; assert(pack_nibbles(16, 1, &out) == -1 && out == 0x55); assert(pack_nibbles(1, 2, NULL) == -1); return 0; }
