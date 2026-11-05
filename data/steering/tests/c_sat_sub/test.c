#include <assert.h>
#include <stdint.h>
int32_t sat_sub(int32_t, int32_t);
int main(void) { assert(sat_sub(9, 4) == 5); assert(sat_sub(INT32_MAX, -1) == INT32_MAX); assert(sat_sub(INT32_MIN, 1) == INT32_MIN); return 0; }
