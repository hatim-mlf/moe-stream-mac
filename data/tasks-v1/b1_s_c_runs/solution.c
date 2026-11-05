#include <stdint.h>
#include <stddef.h>
int encode_runs_u8(const uint8_t *src,size_t n,uint8_t *values,uint32_t *counts,size_t cap,size_t *out_n) {
    if(!out_n) return -1;
    if(!n){*out_n=0;return 0;}
    if(!src||!values||!counts) return -1;
    size_t runs=1;
    for(size_t i=1;i<n;i++) if(src[i]!=src[i-1]) runs++;
    if(runs>cap) return -1;
    size_t r=0;
    for(size_t i=0;i<n;) {
        size_t j=i+1; while(j<n && src[j]==src[i]) j++;
        values[r]=src[i]; counts[r]=(uint32_t)(j-i); r++; i=j;
    }
    *out_n=r; return 0;
}
