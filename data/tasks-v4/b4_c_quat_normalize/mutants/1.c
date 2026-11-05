#include <math.h>
int normalize_quat_f32(const float in[4],float out[4]){if(!in||!out)return -1;double s=0;for(int i=0;i<4;i++)s+=(double)in[i]*in[i];if(s==0)return -1;for(int i=0;i<4;i++)out[i]=(float)(in[i]/sqrt(s));return 0;}
