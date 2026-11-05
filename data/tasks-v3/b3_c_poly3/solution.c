#include <stddef.h>
float cubic_horner_f32(float x,const float c[4]) {
    return ((c[3]*x+c[2])*x+c[1])*x+c[0];
}
