import numpy as np
def nms_xyxy(boxes,scores,threshold):
    b=np.asarray(boxes,float); s=np.asarray(scores,float)
    if b.ndim!=2 or b.shape[1]!=4 or s.shape!=(len(b),): raise ValueError()
    order=np.lexsort((np.arange(len(s)),-s)); kept=[]
    while len(order):
        i=int(order[0]); kept.append(i); rest=[]
        for j in order[1:]:
            inter=max(0,min(b[i,2],b[j,2])-max(b[i,0],b[j,0]))*max(0,min(b[i,3],b[j,3])-max(b[i,1],b[j,1]))
            ai=(b[i,2]-b[i,0])*(b[i,3]-b[i,1]); aj=(b[j,2]-b[j,0])*(b[j,3]-b[j,1]); u=ai+aj-inter
            if (inter/u if u else 0)<=threshold: rest.append(int(j))
        order=np.asarray(rest,dtype=np.int64)
    return np.asarray(kept,dtype=np.int64)
