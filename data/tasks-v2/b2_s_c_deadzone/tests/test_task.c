#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void){
 assert(deadzone_f32(2.5f,1.0f)==1.5f); assert(deadzone_f32(-2.5f,1.0f)==-1.5f);
 assert(deadzone_f32(1.0f,1.0f)==0.0f); assert(deadzone_f32(-0.5f,1.0f)==0.0f);
 assert(deadzone_f32(3.0f,-1.0f)==0.0f); assert(deadzone_f32(INFINITY,1.0f)==0.0f);
 return 0;}
