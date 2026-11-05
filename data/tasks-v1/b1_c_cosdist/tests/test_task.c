#include <assert.h>
#include <math.h>
#include "../candidate.c"
int main(void) {
    float a[]={1,0}, b[]={0,2}, out=7;
    assert(cosine_distance_f32(a,b,2,&out)==0 && fabsf(out-1.0f)<1e-6f);
    float c[]={2,0};
    assert(cosine_distance_f32(a,c,2,&out)==0 && fabsf(out)<1e-6f);
    float z[]={0,0}; out=7;
    assert(cosine_distance_f32(z,c,2,&out)==-1 && out==7);
    assert(cosine_distance_f32(a,b,0,&out)==-1);
}
