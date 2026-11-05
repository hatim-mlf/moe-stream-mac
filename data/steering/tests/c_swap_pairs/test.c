#include <assert.h>
#include <stddef.h>
int swap_pairs(int *, size_t);
int main(void) { int a[] = {1, 2, 3, 4, 5}; assert(swap_pairs(a, 5) == 0); assert(a[0] == 2 && a[1] == 1 && a[2] == 4 && a[3] == 3 && a[4] == 5); assert(swap_pairs(NULL, 0) == 0); assert(swap_pairs(NULL, 1) == -1); return 0; }
