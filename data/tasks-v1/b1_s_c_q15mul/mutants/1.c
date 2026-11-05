#include <stdint.h>
int16_t multiply_q15_round(int16_t a,int16_t b){return (int16_t)(((int32_t)a*b)/32768);}
