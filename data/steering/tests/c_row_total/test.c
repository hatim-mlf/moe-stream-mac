#include <assert.h>
#include <stddef.h>
int row_total(const int *, size_t, size_t, size_t, long long *);
int main(void) { int a[] = {1, 2, 3, 10, 20, 30}; long long out = 0; assert(row_total(a, 2, 3, 1, &out) == 0 && out == 60); assert(row_total(a, 2, 3, 0, &out) == 0 && out == 6); assert(row_total(a, 2, 3, 2, &out) == -1); assert(row_total(NULL, 2, 0, 1, &out) == 0 && out == 0); return 0; }
