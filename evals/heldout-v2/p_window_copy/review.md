Target: candidate.py (Python 3.14; standard library and NumPy only)

Create copied sliding windows along the leading axis.

Exact public signature:
```python
def window_matrix_copy(values, width, step): ...
```

Accept a NumPy-convertible array of rank >=1. `width` and `step` must be integers but not bool; require width>0, step>0, and width<=values.shape[0]. Invalid rank/range raises ValueError; invalid parameter types raise TypeError. Window starts are `0, step, 2*step, ...` while a full width fits. Return a fresh array with shape `(number_of_windows,width)+values.shape[1:]` and the same dtype. Mutating the result must not mutate the input.

Return only the complete file in one fenced python block.

## Your previous attempt

```python
import numpy as np

def window_matrix_copy(values, width, step):
    a=np.asarray(values)
    if a.ndim<1: raise ValueError("values must have a leading axis")
    if isinstance(width,(bool,np.bool_)) or not isinstance(width,(int,np.integer)) or isinstance(step,(bool,np.bool_)) or not isinstance(step,(int,np.integer)):
        raise TypeError("width and step must be integers")
    w,st=int(width),int(step)
    if w<=0 or st<=0 or w>a.shape[0]: raise ValueError("invalid window parameters")
    starts=range(0,a.shape[0]-w,st)
    out=np.empty((len(starts),w)+a.shape[1:],dtype=a.dtype)
    for j,start in enumerate(starts): out[j]=a[start:start+w]
    return out
```

## Review notes

Allocate one output row for every full window, including the final valid start `n-width`. Copy exactly `width` consecutive rows into each window; `step` advances only the start position.
