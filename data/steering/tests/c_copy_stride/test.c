#include <assert.h>
#include <stddef.h>
int copy_stride(const int *, size_t, size_t, int *, size_t, size_t *);
int main(void) { int a[] = {4, 5, 6, 7, 8}; int out[3] = {0}; size_t n = 0; assert(copy_stride(a, 5, 2, out, 3, &n) == 0); assert(n == 3 && out[0] == 4 && out[1] == 6 && out[2] == 8); assert(copy_stride(a, 5, 2, out, 2, &n) == -1); assert(copy_stride(NULL, 0, 2, NULL, 0, &n) == 0 && n == 0); return 0; }
