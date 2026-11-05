#include <stdint.h>
int encode_utf8_scalar(uint32_t c,uint8_t o[4]){if(!o||c>0x10ffffu)return -1;if(c<0x80){o[0]=(uint8_t)c;return 1;}if(c<0x800){o[0]=0xc0|(c>>6);o[1]=0x80|(c&63);return 2;}if(c<0x10000){o[0]=0xe0|(c>>12);o[1]=0x80|((c>>6)&63);o[2]=0x80|(c&63);return 3;}o[0]=0xf0|(c>>18);o[1]=0x80|((c>>12)&63);o[2]=0x80|((c>>6)&63);o[3]=0x80|(c&63);return 4;}
