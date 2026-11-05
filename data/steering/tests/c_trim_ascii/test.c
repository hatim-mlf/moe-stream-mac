#include <assert.h>
#include <stddef.h>
#include <string.h>
char *trim_ascii(char *, size_t);
int main(void) { char s[16] = " \tblue \t"; assert(trim_ascii(s, sizeof s) == s); assert(strcmp(s, "blue") == 0); char line[8] = "\nX\n"; assert(trim_ascii(line, sizeof line) == line); assert(strcmp(line, "\nX\n") == 0); char bad[2] = {'a', 'b'}; assert(trim_ascii(bad, 2) == NULL); return 0; }
