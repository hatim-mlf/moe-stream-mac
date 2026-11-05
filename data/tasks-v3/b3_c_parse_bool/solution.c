#include <string.h>
int parse_ascii_bool(const char *text,int *out) {
    if(text==NULL || out==NULL) return -1;
    int value;
    if(strcmp(text,"true")==0 || strcmp(text,"1")==0) value=1;
    else if(strcmp(text,"false")==0 || strcmp(text,"0")==0) value=0;
    else return -1;
    *out=value;
    return 0;
}
