#include <assert.h>
#include <stddef.h>
int rotate_left(int *, size_t, size_t);
int main(void) { int a[] = {1, 2, 3, 4, 5}; assert(rotate_left(a, 5, 7) == 0); assert(a[0] == 3 && a[1] == 4 && a[2] == 5 && a[3] == 1 && a[4] == 2); assert(rotate_left(NULL, 0, 3) == 0); assert(rotate_left(NULL, 2, 1) == -1); return 0; }
