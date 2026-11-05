#include <assert.h>
#include <stdint.h>
#include "../candidate.c"
int main(void) {
    TinyDeque q; int32_t x=99; tiny_deque_init(&q);
    assert(tiny_deque_push_back(&q,1)==0 && tiny_deque_push_front(&q,2)==0);
    assert(tiny_deque_pop_back(&q,&x)==0 && x==1);
    assert(tiny_deque_pop_front(&q,&x)==0 && x==2);
    for(int i=0;i<8;++i) assert(tiny_deque_push_back(&q,i)==0);
    assert(tiny_deque_push_front(&q,9)==-1 && q.size==8);
    assert(tiny_deque_pop_front(&q,&x)==0 && x==0);
    assert(tiny_deque_pop_back(&q,&x)==0 && x==7);
    return 0;
}
