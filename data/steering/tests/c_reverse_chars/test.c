#include <assert.h>
#include <stddef.h>
#include <string.h>
int reverse_chars(char *, size_t);
int main(void) { char a[8] = "cabin"; assert(reverse_chars(a, sizeof a) == 0); assert(strcmp(a, "nibac") == 0); char empty[1] = {0}; assert(reverse_chars(empty, 1) == 0); char bad[2] = {'a', 'b'}; assert(reverse_chars(bad, 2) == -1); assert(reverse_chars(NULL, 1) == -1); return 0; }
