#include <assert.h>
#include <stddef.h>
#include <string.h>
int join_two(const char *, const char *, char, char *, size_t, size_t *);
int main(void) { char out[16] = {0}; size_t n = 0; assert(join_two("red", "blue", '/', out, sizeof out, &n) == 0); assert(strcmp(out, "red/blue") == 0 && n == 8); char small[8] = "keep"; assert(join_two("red", "blue", '/', small, sizeof small, &n) == -1); assert(strcmp(small, "keep") == 0); return 0; }
