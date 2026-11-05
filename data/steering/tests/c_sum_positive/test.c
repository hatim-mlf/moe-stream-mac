#include <assert.h>
#include <stddef.h>
long long sum_positive(const int *, size_t);
int main(void) { int a[] = {-3, 0, 4, 7, -1}; assert(sum_positive(a, 5) == 11); assert(sum_positive(NULL, 0) == 0); assert(sum_positive(NULL, 2) == 0); return 0; }
