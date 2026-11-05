#include <stddef.h>
int is_ascii_identifier(const char*t){if(!t||!*t)return 0;for(size_t i=0;t[i];i++){unsigned char c=(unsigned char)t[i];int alpha=(c>='A'&&c<='Z')||(c>='a'&&c<='z');if(i==0){if(!alpha&&c!='_'&&!(c>='0'&&c<='9'))return 0;}else if(!alpha&&!(c>='0'&&c<='9')&&c!='_')return 0;}return 1;}
