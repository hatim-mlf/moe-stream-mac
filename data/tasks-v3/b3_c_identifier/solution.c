#include <stddef.h>
int is_ascii_identifier(const char *text) {
    if(text==0 || *text==0) return 0;
    for(size_t i=0;text[i];i++) {
        unsigned char c=(unsigned char)text[i];
        int letter=(c>='A'&&c<='Z') || (c>='a'&&c<='z');
        if(i==0) { if(!letter && c!='_') return 0; }
        else if(!letter && !(c>='0'&&c<='9') && c!='_') return 0;
    }
    return 1;
}
