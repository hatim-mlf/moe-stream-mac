#include <stdint.h>
#include <stddef.h>
int parse_i16_csv(const char*t,int16_t*out,size_t cap,size_t*count){if(!t||!count)return -1;size_t n=0;const char*p=t;while(*p){if(n>=cap||!out)return -1;int neg=0,v=0;if(*p=='-'||*p=='+'){neg=*p=='-';p++;}if(*p<'0'||*p>'9')return -1;while(*p>='0'&&*p<='9'){v=v*10+*p-'0';p++;}out[n++]=(int16_t)(neg?-v:v);if(*p==',')p++;else if(*p)return -1;}*count=n;return (int)n;}
