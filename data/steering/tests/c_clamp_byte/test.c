#include <assert.h>
int clamp_byte(int value);
int main(void) { assert(clamp_byte(-4) == 0); assert(clamp_byte(0) == 0); assert(clamp_byte(255) == 255); assert(clamp_byte(300) == 255); return 0; }
