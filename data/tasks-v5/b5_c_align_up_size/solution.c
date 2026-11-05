#include <stddef.h>
#include <stdint.h>
int align_up_size(size_t value,size_t alignment,size_t*out){if(!out||!alignment)return -1;size_t rem=value%alignment;if(rem&&value>SIZE_MAX-(alignment-rem))return -1;*out=rem?value+(alignment-rem):value;return 0;}
