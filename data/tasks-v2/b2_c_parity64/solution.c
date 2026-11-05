#include <stdint.h>
unsigned parity_u64(uint64_t value) {
    unsigned parity=0;
    while(value){parity^=1u;value&=value-1u;}
    return parity;
}
