#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
int parse_i32_pair(const char *text,int32_t *left,int32_t *right) {
    if(!text||!left||!right) return -1;
    if(!((*text>='0'&&*text<='9')||*text=='+'||*text=='-')) return -1;
    errno=0; char *end1;
    long a=strtol(text,&end1,10);
    if(errno==ERANGE || end1==text || *end1!=',') return -1;
    const char *second=end1+1;
    if(!((*second>='0'&&*second<='9')||*second=='+'||*second=='-')) return -1;
    errno=0; char *end2;
    long b=strtol(second,&end2,10);
    if(errno==ERANGE || end2==second || *end2!='\0' ||
       a<INT32_MIN || a>INT32_MAX || b<INT32_MIN || b>INT32_MAX) return -1;
    *left=(int32_t)a; *right=(int32_t)b;
    return 0;
}
