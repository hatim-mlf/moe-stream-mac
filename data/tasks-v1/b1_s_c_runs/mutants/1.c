#include <stdint.h>
#include <stddef.h>
int encode_runs_u8(const uint8_t*s,size_t n,uint8_t*v,uint32_t*c,size_t cap,size_t*out){
    if(!out||(n&&(!s||!v||!c)))return -1;
    size_t r=0;
    for(size_t i=0;i<n;i++){if(i==0||s[i]!=s[i-1]){if(r>=cap)return -1;v[r]=s[i];c[r]=0;r++;}c[r-1]++;}
    *out=r;return 0;
}
