#include <stdint.h>
#include <stdio.h>
int parse_i32_pair(const char *text,int32_t *left,int32_t *right) {
    int a,b;
    if(!text||!left||!right || sscanf(text,"%d,%d",&a,&b)!=2) return -1;
    *left=a; *right=b; return 0;
}
