#include <assert.h>
#include <stddef.h>
#include <stdint.h>
int xor_bytes(const uint8_t *, const uint8_t *, uint8_t *, size_t);
int main(void) { uint8_t a[] = {0x0f, 0xf0, 0x55}; uint8_t b[] = {0xf0, 0x0f, 0x55}; uint8_t out[3]; assert(xor_bytes(a, b, out, 3) == 0); assert(out[0] == 0xff && out[1] == 0xff && out[2] == 0); assert(xor_bytes(a, b, a, 3) == 0 && a[0] == 0xff); assert(xor_bytes(NULL, NULL, NULL, 0) == 0); assert(xor_bytes(NULL, b, out, 1) == -1); return 0; }
