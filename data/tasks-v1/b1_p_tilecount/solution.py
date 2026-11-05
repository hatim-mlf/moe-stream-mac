import numpy as np
def count_true_tiles(mask,tile_rows,tile_cols):
    a=np.asarray(mask)
    if a.ndim!=2 or a.dtype!=np.bool_ or not isinstance(tile_rows,(int,np.integer)) or not isinstance(tile_cols,(int,np.integer)) or tile_rows<=0 or tile_cols<=0:
        raise ValueError("invalid mask or tile shape")
    nr=(a.shape[0]+tile_rows-1)//tile_rows
    nc=(a.shape[1]+tile_cols-1)//tile_cols
    out=np.zeros((nr,nc),dtype=np.int64)
    for r in range(nr):
        for c in range(nc):
            block=a[r*tile_rows:min((r+1)*tile_rows,a.shape[0]),c*tile_cols:min((c+1)*tile_cols,a.shape[1])]
            out[r,c]=np.count_nonzero(block)
    return out
