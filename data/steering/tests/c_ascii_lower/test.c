#include <assert.h>
#include <stddef.h>
int ascii_lower(char *, size_t);
int main(void) { char s[8] = {'A', '[', 'z', 0}; assert(ascii_lower(s, sizeof s) == 1); assert(s[0] == 'a' && s[1] == '[' && s[2] == 'z'); char bad[2] = {'A', 'B'}; assert(ascii_lower(bad, 2) == -1); return 0; }
