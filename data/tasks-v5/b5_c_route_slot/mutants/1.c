#include <stdint.h>
#include <stddef.h>
int parse_route_slot(const char*t,char*n,size_t c,uint16_t*s){
 if(!t||!n||!s)return -1; size_t k=0; while(t[k]&&t[k]!='@'){unsigned char x=(unsigned char)t[k];if((k==0&&(x<'a'||x>'z'))||(k>0&&!((x>='a'&&x<='z')||(x>='0'&&x<='9')||x=='_')))return -1;++k;}
 if(!k||t[k]!='@'||c<=k)return -1;const char*p=t+k+1;if(!*p)return -1;unsigned v=0;for(;*p;++p){if(*p<'0'||*p>'9')return -1;v=v*10u+(unsigned)(*p-'0');if(v>65535)return -1;}for(size_t i=0;i<k;++i)n[i]=t[i];n[k]='\0';*s=(uint16_t)v;return 0;
}
