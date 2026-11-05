#include <assert.h>
int is_ascii_hex(int value);
int main(void) { assert(is_ascii_hex('0') == 1); assert(is_ascii_hex('f') == 1); assert(is_ascii_hex('G') == 0); assert(is_ascii_hex('z') == 0); assert(is_ascii_hex(-1) == 0); return 0; }
