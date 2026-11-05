#include <string.h>
int parse_ascii_bool(const char*t,int*out){if(!t||!out)return -1;*out=(t[0]=='t'||t[0]=='1');return 0;}
