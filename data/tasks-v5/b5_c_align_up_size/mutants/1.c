#include <stddef.h>
#include <stdint.h>
int align_up_size(size_t v,size_t a,size_t*out){if(!out||!a)return -1;size_t r=v%a;if(r)*out=v+(a-r);else *out=v;return 0;}
