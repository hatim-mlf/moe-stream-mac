#include <stdint.h>
unsigned parity_u64(uint64_t x){unsigned p=0;while(x){p^=1u;x>>=2;}return p;}
