#include <assert.h>
#include <stddef.h>
#include <string.h>
int replace_first(char *, size_t, char, char);
int main(void) { char s[8] = "banana"; assert(replace_first(s, sizeof s, 'a', 'o') == 1); assert(strcmp(s, "bonana") == 0); assert(replace_first(s, sizeof s, 'z', 'x') == 0); char bad[2] = {'a', 'b'}; assert(replace_first(bad, 2, 'a', 'x') == -1); return 0; }
