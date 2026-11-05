#include <stddef.h>
#include <stdint.h>
typedef struct { uint16_t tag; uint8_t size; uint8_t bytes[4]; } Packet;
typedef struct { Packet incoming[4]; Packet outgoing[4]; size_t in_count; size_t out_count; } PacketQueue;
void packet_queue_init(PacketQueue*q){if(q){q->in_count=0;q->out_count=0;}}
int packet_queue_push(PacketQueue*q,const Packet*p){if(!q||!p||p->size>4||q->in_count+q->out_count>=4)return -1;q->incoming[q->in_count++]=*p;return 0;}
int packet_queue_pop(PacketQueue*q,Packet*out){if(!q||!out||q->in_count+q->out_count==0)return -1;if(!q->out_count){for(size_t i=0;i<q->in_count;i++)q->outgoing[q->out_count++]=q->incoming[i];q->in_count=0;}*out=q->outgoing[--q->out_count];return 0;}