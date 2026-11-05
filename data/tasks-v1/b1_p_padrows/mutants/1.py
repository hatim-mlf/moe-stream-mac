import numpy as np
def pad_numeric_rows(rows,pad_value=0):
    arrays=[np.asarray(row) for row in rows]
    width=max((len(a) for a in arrays),default=0)
    matrix=np.full((len(arrays),width),pad_value)
    for i,a in enumerate(arrays): matrix[i,:len(a)]=a
    return matrix,np.asarray([len(a) for a in arrays])
