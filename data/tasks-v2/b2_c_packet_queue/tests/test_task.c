#include <assert.h>
#include "../candidate.c"
int main(void){PacketQueue q;Packet a={11,2,{4,5,0,0}},b={22,1,{9,0,0,0}},c={33,1,{8,0,0,0}},out={77,1,{7,0,0,0}};
 packet_queue_init(&q);
 assert(packet_queue_pop(&q,&out)==-1&&out.tag==77);
 assert(packet_queue_push(&q,&a)==0&&packet_queue_push(&q,&b)==0);
 assert(packet_queue_pop(&q,&out)==0&&out.tag==11&&out.bytes[1]==5);
 assert(packet_queue_push(&q,&c)==0&&packet_queue_push(&q,&a)==0&&packet_queue_push(&q,&b)==0);
 assert(packet_queue_push(&q,&c)==-1&&q.in_count+q.out_count==4);
 assert(packet_queue_pop(&q,&out)==0&&out.tag==22);
 assert(packet_queue_pop(&q,&out)==0&&out.tag==33);
 assert(packet_queue_pop(&q,&out)==0&&out.tag==11);
 assert(packet_queue_pop(&q,&out)==0&&out.tag==22);
 a.size=5;assert(packet_queue_push(&q,&a)==-1&&q.in_count+q.out_count==0);
 return 0;
}