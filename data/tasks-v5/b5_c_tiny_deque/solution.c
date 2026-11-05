#include <stdint.h>
#include <stddef.h>
typedef struct { int32_t data[8]; size_t head, size; } TinyDeque;
void tiny_deque_init(TinyDeque *q) { if (q) { q->head=0; q->size=0; } }
int tiny_deque_push_front(TinyDeque *q,int32_t v) {
    if(!q||q->size>=8)return -1; q->head=(q->head+7)%8; q->data[q->head]=v; ++q->size; return 0;
}
int tiny_deque_push_back(TinyDeque *q,int32_t v) {
    if(!q||q->size>=8)return -1; q->data[(q->head+q->size)%8]=v; ++q->size; return 0;
}
int tiny_deque_pop_front(TinyDeque *q,int32_t *out) {
    if(!q||!out||!q->size)return -1; *out=q->data[q->head]; q->head=(q->head+1)%8; --q->size; return 0;
}
int tiny_deque_pop_back(TinyDeque *q,int32_t *out) {
    if(!q||!out||!q->size)return -1; *out=q->data[(q->head+q->size-1)%8]; --q->size; return 0;
}
