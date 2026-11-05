#include <stddef.h>
#include <stdint.h>
typedef struct { uint8_t *data; size_t capacity, head, size; } ByteRing;
void byte_ring_init(ByteRing *q,uint8_t *s,size_t c){q->data=s;q->capacity=c;q->head=0;q->size=0;}
int byte_ring_write(ByteRing *q,const uint8_t *src,size_t n){if(!q||!q->data||(n&&!src)||n>q->capacity-q->size)return -1;for(size_t i=0;i<n;i++)q->data[(q->head+q->size+i)%q->capacity]=src[i];q->size+=n;return 0;}
int byte_ring_read(ByteRing *q,uint8_t *dst,size_t n){if(!q||!q->data||(n&&!dst)||n>q->size)return -1;for(size_t i=0;i<n;i++)dst[i]=q->data[(q->head+i)%q->capacity];q->head=(q->head+n)%q->capacity;q->size-=n;return 0;}
