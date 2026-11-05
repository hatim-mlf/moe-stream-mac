#include <assert.h>
#include <stddef.h>
#include <stdint.h>
int parse_u16(const char *, uint16_t *);
int main(void) { uint16_t out = 77; assert(parse_u16("65535", &out) == 1 && out == 65535); out = 77; assert(parse_u16("65536", &out) == 0 && out == 77); assert(parse_u16("12x", &out) == 0 && out == 77); assert(parse_u16("+2", &out) == 0 && out == 77); return 0; }
