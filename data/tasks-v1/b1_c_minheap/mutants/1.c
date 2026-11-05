#include <stddef.h>
typedef struct { int data[16]; size_t len; } SmallMinHeap;
void minheap_init(SmallMinHeap *q){if(q)q->len=0;}
int minheap_push(SmallMinHeap *q,int value){
    if(!q||q->len>=16)return -1;
    size_t i=q->len++;
    while(i>0){size_t p=(i-1)/2;if(q->data[p]>=value)break;q->data[i]=q->data[p];i=p;}
    q->data[i]=value;return 0;
}
int minheap_pop(SmallMinHeap *q,int *out){
    if(!q||!out||!q->len)return -1;
    *out=q->data[0];q->data[0]=q->data[--q->len];return 0;
}
