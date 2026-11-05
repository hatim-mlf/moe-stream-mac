#include <assert.h>
#include <stddef.h>
size_t count_spaces_ascii(const unsigned char *, size_t);
int main(void) { const unsigned char a[] = {' ', '\t', '\n', 'x', ' '}; assert(count_spaces_ascii(a, sizeof a) == 2); assert(count_spaces_ascii(NULL, 3) == 0); return 0; }
