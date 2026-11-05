int ascii_casecmp(const char*a,const char*b){for(;;){unsigned char x=(unsigned char)*a++,y=(unsigned char)*b++;if(x!=y)return x<y?-1:1;if(!x)return 0;}}
