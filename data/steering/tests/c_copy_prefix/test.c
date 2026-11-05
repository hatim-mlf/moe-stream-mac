#include <assert.h>
#include <stddef.h>
#include <string.h>
int copy_prefix(const char *, size_t, char *, size_t, size_t *);
int main(void) { char out[8] = {0}; size_t n = 0; assert(copy_prefix("planet", 4, out, sizeof out, &n) == 0); assert(n == 4 && strcmp(out, "plan") == 0); assert(copy_prefix("abc", 3, out, 3, &n) == -1); assert(copy_prefix("abc", 3, out, sizeof out, NULL) == -1); return 0; }
