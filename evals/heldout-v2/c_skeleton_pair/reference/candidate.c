#include <stdint.h>
int parse_pair_record(const char *text, int16_t *left, int16_t *right) {
    /* TODO_BEGIN */
    const char *p=text; int values[2], which;
    if (!text || !left || !right) return -1;
    for (which=0;which<2;++which) {
        int neg=0; unsigned long v=0; unsigned long limit;
        if (*p=='+' || *p=='-') { neg=(*p=='-'); ++p; }
        if (*p<'0' || *p>'9') return -2;
        limit=neg ? 32768ul : 32767ul;
        while (*p>='0' && *p<='9') { unsigned d=(unsigned)(*p-'0'); if (v>(limit-d)/10ul) return -3; v=v*10ul+d; ++p; }
        values[which]=neg ? -(int)v : (int)v;
        if (which==0) { if (*p++!=',') return -2; }
        else if (*p!='\0') return -2;
    }
    *left=(int16_t)values[0]; *right=(int16_t)values[1]; return 0;
    /* TODO_END */
}
