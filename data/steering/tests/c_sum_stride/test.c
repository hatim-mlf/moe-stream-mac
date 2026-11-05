#include <assert.h>
#include <stddef.h>
long long sum_stride(const int *, size_t, size_t);
int main(void) { int a[] = {2, 10, 3, 20, 4, 30}; assert(sum_stride(a, 6, 2) == 9); assert(sum_stride(a, 6, 3) == 22); assert(sum_stride(NULL, 0, 1) == 0); assert(sum_stride(a, 6, 0) == -1); return 0; }
