#include <stdint.h>
#include <stddef.h>
int interleave_i16(const int16_t*a,const int16_t*b,size_t n,int16_t*out,size_t cap){
    if(n>cap/2 || (n&&(!a||!b||!out)))return -1;
    for(size_t i=0;i<n;i++){out[2*i]=b[i];out[2*i+1]=a[i];}
    return 0;
}
