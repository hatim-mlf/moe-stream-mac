#include <stdint.h>
int16_t multiply_q15_round(int16_t a,int16_t b) {
    int64_t p=(int64_t)a*b;
    uint64_t mag=(uint64_t)(p<0 ? -p : p);
    int64_t q=(int64_t)((mag+16384u)/32768u);
    if(p<0) q=-q;
    if(q>32767) q=32767;
    if(q< -32768) q=-32768;
    return (int16_t)q;
}
