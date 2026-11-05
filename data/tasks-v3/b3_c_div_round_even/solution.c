#include <stdint.h>
#include <stddef.h>
#include <limits.h>
int div_round_even_i64(int64_t numerator,int64_t denominator,int64_t *out) {
    if (out==NULL || denominator==0 || (numerator==INT64_MIN && denominator==-1)) return -1;
    int64_t q=numerator/denominator, r=numerator%denominator;
    uint64_t ar=r<0 ? (uint64_t)(-(r+1))+1u : (uint64_t)r;
    uint64_t ad=denominator<0 ? (uint64_t)(-(denominator+1))+1u : (uint64_t)denominator;
    int away=(ar > ad-ar) || ((ar == ad-ar) && (q % 2 != 0));
    if (away) {
        int positive=(numerator<0)==(denominator<0);
        if ((positive && q==INT64_MAX) || (!positive && q==INT64_MIN)) return -1;
        q += positive ? 1 : -1;
    }
    *out=q;
    return 0;
}
