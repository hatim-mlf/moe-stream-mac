import numpy as np
def project_homography2d(points,matrix):
    p=np.asarray(points,dtype=float); m=np.asarray(matrix,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or m.shape!=(3,3): raise ValueError()
    h=np.column_stack((p,np.ones(len(p))))@m.T
    if np.any(h[:,2]==0): raise ValueError()
    return h[:,:2]
