#include <stdint.h>
#include <stdio.h>
int parse_i32_pair(const char *text,int32_t *left,int32_t *right) {
    int a,b; char extra;
    if(!text||!left||!right || sscanf(text,"%d,%d%c",&a,&b,&extra)<2) return -1;
    *left=a; *right=b; return 0;
}
