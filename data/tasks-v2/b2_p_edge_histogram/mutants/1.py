import numpy as np
def histogram_with_edges(values,edges):
 x=np.asarray(values,dtype=np.float64); e=np.asarray(edges,dtype=np.float64)
 if x.ndim!=1 or e.ndim!=1 or e.size<2 or not np.isfinite(x).all() or not np.isfinite(e).all() or np.any(e[1:]<=e[:-1]): raise ValueError()
 bins=np.searchsorted(e,x,side='right')-1; keep=(x>=e[0])&(x<e[-1])
 return np.bincount(bins[keep],minlength=e.size-1).astype(np.int64)
