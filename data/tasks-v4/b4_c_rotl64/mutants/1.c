#include <stdint.h>
uint64_t rotate_left_u64(uint64_t value,unsigned shift){shift%=63u;if(!shift)return value;return (value<<shift)|(value>>(64u-shift));}
