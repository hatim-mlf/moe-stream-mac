import math
import numbers

def format_percent_1dp(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError('value must be a finite real scalar')
    return f'{value * 100:.1f}%'
