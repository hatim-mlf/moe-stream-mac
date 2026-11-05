#include <stdint.h>
int parse_version_triplet(const char *s,uint16_t out[3]){unsigned v[3]={0};if(!s||!out||*s++!='v')return -1;for(int f=0;f<3;f++){if(*s<'0'||*s>'9')return -1;do{v[f]=v[f]*10u+(unsigned)(*s-'0');if(v[f]>65535u)return -1;s++;}while(*s>='0'&&*s<='9');if(f<2){if(*s++!='.')return -1;}else if(*s)return -1;}for(int i=0;i<3;i++)out[i]=(uint16_t)v[i];return 0;}
