import math
import numbers
def format_percent_1dp(value):
    if isinstance(value,bool) or not isinstance(value,numbers.Real) or not math.isfinite(value): raise ValueError()
    return f'{value:.0%}'
