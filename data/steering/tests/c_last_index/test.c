#include <assert.h>
#include <stddef.h>
long last_index(const int *, size_t, int);
int main(void) { int a[] = {4, 9, 4, 2, 4}; assert(last_index(a, 5, 4) == 4); assert(last_index(a, 5, 8) == -1); assert(last_index(NULL, 1, 4) == -1); return 0; }
