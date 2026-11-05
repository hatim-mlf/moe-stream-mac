#include <stddef.h>
#include <stdint.h>
int copy_byte_span(const uint8_t *src,size_t src_len,size_t offset,uint8_t *dst,size_t dst_cap,size_t count){if(offset>src_len||count>src_len-offset||count>dst_cap||(count&&(!src||!dst)))return -1;for(size_t i=0;i<count;i++)dst[i]=src[offset+i];return 0;}
