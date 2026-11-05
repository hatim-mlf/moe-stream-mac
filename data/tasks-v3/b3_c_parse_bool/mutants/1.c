#include <string.h>
int parse_ascii_bool(const char*t,int*out){if(!t||!out)return -1;if(strcmp(t,"true")==0||strcmp(t,"True")==0||strcmp(t,"1")==0){*out=1;return 0;}if(strcmp(t,"false")==0||strcmp(t,"False")==0||strcmp(t,"0")==0){*out=0;return 0;}return -1;}
