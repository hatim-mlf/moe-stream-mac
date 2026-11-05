#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
int format_hex(uint8_t, char *, size_t);
int main(void) { char out[4] = {'x', 'x', 'x', 'x'}; assert(format_hex(0xaf, out, sizeof out) == 2); assert(strcmp(out, "AF") == 0); char small[2] = {'q', 'q'}; assert(format_hex(1, small, sizeof small) == -1 && small[0] == 'q'); return 0; }
