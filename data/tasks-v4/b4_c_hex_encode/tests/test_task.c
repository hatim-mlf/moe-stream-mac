#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "../candidate.c"
int main(void) {
    const uint8_t src[] = {0x00, 0x09, 0xaf, 0xff}; char out[9] = "XXXXXXXX";
    assert(hex_encode_lower(src, 4, out, sizeof out) == 8);
    assert(strcmp(out, "0009afff") == 0);
    strcpy(out, "XXXXXXXX"); assert(hex_encode_lower(src, 4, out, 8) == -1);
    assert(strcmp(out, "XXXXXXXX") == 0);
    assert(hex_encode_lower(NULL, 0, out, 1) == 0 && out[0] == '\0');
    return 0;
}
