#include <assert.h>
#include <stddef.h>
size_t count_ascii_digits(const unsigned char *, size_t);
int main(void) { const unsigned char a[] = {'0', 'a', '9', '/', 0xff, '4'}; assert(count_ascii_digits(a, sizeof a) == 3); assert(count_ascii_digits(NULL, 7) == 0); return 0; }
