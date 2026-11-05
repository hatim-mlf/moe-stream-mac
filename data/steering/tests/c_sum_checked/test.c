#include <assert.h>
#include <stddef.h>
int sum_checked(const int *, size_t, long long *);
int main(void) { int a[] = {5, -2, 8}; long long out = -1; assert(sum_checked(a, 3, &out) == 0 && out == 11); assert(sum_checked(NULL, 0, &out) == 0 && out == 0); assert(sum_checked(NULL, 1, &out) == -1); assert(sum_checked(a, 3, NULL) == -1); return 0; }
