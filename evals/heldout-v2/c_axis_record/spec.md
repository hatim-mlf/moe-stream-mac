Target: candidate.c (C99)

Parse a compact axis and half-open range record.

Exact signature:
```c
int parse_axis_record(const char *text, int *axis, int *start, int *end);
```

Grammar is exactly `A<axis>,<start>,<end>` with unsigned decimal digits and no whitespace or trailing characters. Axis must be 0..7 and `0 <= start < end`. Return 0 on success, -1 for null arguments, -2 for syntax errors, and -3 for numeric overflow or out-of-range values. Assign all three outputs only on success.

Return only the complete file in one fenced c block.
