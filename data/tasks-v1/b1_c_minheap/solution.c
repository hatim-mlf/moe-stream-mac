#include <stddef.h>
typedef struct { int data[16]; size_t len; } SmallMinHeap;
void minheap_init(SmallMinHeap *q) { if (q) q->len=0; }
int minheap_push(SmallMinHeap *q,int value) {
    if (!q || q->len>=16) return -1;
    size_t i=q->len++;
    while (i>0) {
        size_t p=(i-1)/2;
        if (q->data[p]<=value) break;
        q->data[i]=q->data[p]; i=p;
    }
    q->data[i]=value;
    return 0;
}
int minheap_pop(SmallMinHeap *q,int *out) {
    if (!q || !out || q->len==0) return -1;
    *out=q->data[0];
    int last=q->data[--q->len];
    if (q->len) {
        size_t i=0;
        while (2*i+1<q->len) {
            size_t child=2*i+1;
            if (child+1<q->len && q->data[child+1]<q->data[child]) ++child;
            if (q->data[child]>=last) break;
            q->data[i]=q->data[child]; i=child;
        }
        q->data[i]=last;
    }
    return 0;
}
