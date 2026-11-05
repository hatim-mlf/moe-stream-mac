#include <stdint.h>
int lcm_u32(uint32_t a,uint32_t b,uint32_t *o){if(!o)return -1;if(!a||!b){*o=0;return 0;}uint32_t x=a,y=b;while(y){uint32_t r=x%y;x=y;y=r;}*o=(a/x)*b;return 0;}
