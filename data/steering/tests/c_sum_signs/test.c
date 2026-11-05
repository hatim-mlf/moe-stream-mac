#include <assert.h>
#include <stddef.h>
int sum_signs(const int *, size_t, long long *, long long *);
int main(void) { int a[] = {-5, 0, 7, -2, 3}; long long p = 0, n = 0; assert(sum_signs(a, 5, &p, &n) == 0); assert(p == 10 && n == -7); assert(sum_signs(NULL, 0, &p, &n) == 0 && p == 0 && n == 0); assert(sum_signs(a, 5, NULL, &n) == -1); return 0; }
