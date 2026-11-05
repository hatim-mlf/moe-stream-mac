#include <assert.h>
#include <stddef.h>
long first_nonzero(const unsigned char *, size_t);
int main(void) { const unsigned char a[] = {0, 0, 4, 0, 9}; const unsigned char z[] = {0, 0}; assert(first_nonzero(a, 5) == 2); assert(first_nonzero(z, 2) == -1); assert(first_nonzero(NULL, 2) == -1); return 0; }
