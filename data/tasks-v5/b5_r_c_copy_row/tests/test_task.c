#include <assert.h>
#include "../candidate.c"
int main(void){float m[]={1,2,3,4,5,6},o[]={9,9,9};assert(copy_matrix_row_f32(m,2,3,1,o,3)==0&&o[0]==4&&o[2]==6);o[0]=9;assert(copy_matrix_row_f32(m,2,3,2,o,3)==-1&&o[0]==9);assert(copy_matrix_row_f32(NULL,2,0,1,NULL,0)==0);return 0;}
