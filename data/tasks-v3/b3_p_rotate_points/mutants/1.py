import numpy as np
def rotate_points2d(points,angle):
 p=np.asarray(points,dtype=float); c=np.cos(angle); s=np.sin(angle)
 return np.stack((c*p[...,0]-s*p[...,1],s*p[...,0]+c*p[...,1]),axis=-1)
