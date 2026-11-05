#include <stddef.h>
#include <stdint.h>
int align_up_size(size_t v,size_t a,size_t*out){if(!out||!a)return -1;*out=(v+a-1)&~(a-1);return 0;}
