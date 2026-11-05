#include <stdint.h>
int parse_hex_u32_exact(const char*t,uint32_t*out){if(!t||!out)return -1;uint32_t v=0;unsigned n=0;for(;t[n];n++){unsigned char c=(unsigned char)t[n];unsigned d;if(c>='0'&&c<='9')d=c-'0';else if(c>='a'&&c<='f')d=c-'a'+10;else if(c>='A'&&c<='F')d=c-'A'+10;else return -1;v=(v<<4)|d;}if(n==0)return -1;*out=v;return 0;}
