#include <assert.h>
#include "../candidate.c"
int main(void){assert(ascii_casecmp("Gate","gATE")==0);assert(ascii_casecmp("a","B")==-1);assert(ascii_casecmp("Z","y")==1);assert(ascii_casecmp("a","aa")==-1);return 0;}
