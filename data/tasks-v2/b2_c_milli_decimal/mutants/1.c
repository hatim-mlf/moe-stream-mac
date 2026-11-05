#include <stdint.h>
#include <limits.h>
int parse_milli_i64(const char*t,int64_t*out){if(!t||!out)return -1;int neg=0;if(*t=='+'||*t=='-'){neg=*t=='-';t++;}if(*t<'0'||*t>'9')return -1;uint64_t w=0;while(*t>='0'&&*t<='9'){w=w*10u+(unsigned)(*t++-'0');}unsigned f=0,n=0;if(*t=='.'){t++;while(*t>='0'&&*t<='9'){if(n==3)return -1;f=f*10u+(unsigned)(*t++-'0');n++;}if(!n)return -1;}if(*t)return -1;while(n++<3)f*=10u;uint64_t m=w*1000u+f;*out=neg?-(int64_t)m:(int64_t)m;return 0;}
