import numpy as np

def calibrate_scores(raw, temperature, offset):
    # TODO_BEGIN
    a=np.asarray(raw)
    if a.dtype.kind not in "iuf" or not np.all(np.isfinite(a)):
        raise ValueError("raw scores must be finite real values")
    if not np.isscalar(temperature) or not np.isscalar(offset):
        raise TypeError("temperature and offset must be scalars")
    if not np.isfinite(temperature) or not np.isfinite(offset) or temperature<=0:
        raise ValueError("invalid calibration parameters")
    z=(a.astype(np.float64)-float(offset))/float(temperature)
    out=np.empty(z.shape,dtype=np.float64)
    positive=z>=0
    out[positive]=1.0/(1.0+np.exp(-z[positive]))
    ez=np.exp(z[~positive])
    out[~positive]=ez/(1.0+ez)
    return out.astype(np.float32)
    # TODO_END
