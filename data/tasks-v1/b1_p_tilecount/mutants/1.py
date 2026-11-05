import numpy as np
def count_true_tiles(mask,tile_rows,tile_cols):
    a=np.asarray(mask,dtype=bool)
    nr=a.shape[0]//tile_rows; nc=a.shape[1]//tile_cols
    out=np.zeros((nr,nc),dtype=np.int64)
    for r in range(nr):
        for c in range(nc):
            out[r,c]=a[r*tile_rows:(r+1)*tile_rows,c*tile_cols:(c+1)*tile_cols].sum()
    return out
