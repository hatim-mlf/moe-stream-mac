import numpy as np

def sliding_windows_1d(values,width,step=1):
    a=np.asarray(values)
    if a.ndim!=1 or width<=0 or step<=0: raise ValueError
    return np.lib.stride_tricks.sliding_window_view(a,width)[::step]
