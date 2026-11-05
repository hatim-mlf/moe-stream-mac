Target: candidate.c (C99)

Fill the signed 16-bit pair-record parser.

Complete file template (replace only each TODO body; keep all other lines unchanged):
```c
#include <stdint.h>
int parse_pair_record(const char *text, int16_t *left, int16_t *right) {
    /* TODO_BEGIN */
    /* TODO_END */
}
```

Grammar is two signed decimal integers separated by one comma, with optional single leading + or - on each number and no whitespace/trailing text. Both values must fit int16. Return 0 on success, -1 for null arguments, -2 for malformed syntax, and -3 for out-of-range values. Assign outputs only on success.

Return only the complete file in one fenced c block.
