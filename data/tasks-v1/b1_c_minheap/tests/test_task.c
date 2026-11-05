#include <assert.h>
#include "../candidate.c"
int main(void) {
    SmallMinHeap q; int v=99;
    minheap_init(&q);
    assert(minheap_pop(&q,&v)==-1 && v==99);
    assert(minheap_push(&q,7)==0 && minheap_push(&q,-2)==0);
    assert(minheap_push(&q,4)==0 && minheap_push(&q,-2)==0);
    assert(minheap_pop(&q,&v)==0 && v==-2);
    assert(minheap_pop(&q,&v)==0 && v==-2);
    assert(minheap_pop(&q,&v)==0 && v==4);
    assert(minheap_pop(&q,&v)==0 && v==7);
    for(int i=0;i<16;i++) assert(minheap_push(&q,i)==0);
    assert(minheap_push(&q,17)==-1 && q.len==16);
}
