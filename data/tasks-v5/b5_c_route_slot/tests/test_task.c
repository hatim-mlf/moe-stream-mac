#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include "../candidate.c"
int main(void) {
    char name[8]="keep"; uint16_t slot=9;
    assert(parse_route_slot("gate_2@65535",name,sizeof name,&slot)==0 && slot==65535 && name[6]=='\0');
    assert(parse_route_slot("a@0",name,sizeof name,&slot)==0 && slot==0 && name[0]=='a');
    slot=44; name[0]='Z'; assert(parse_route_slot("a@01",name,sizeof name,&slot)==-1 && slot==44 && name[0]=='Z');
    assert(parse_route_slot("a@65536",name,sizeof name,&slot)==-1); return 0;
}
