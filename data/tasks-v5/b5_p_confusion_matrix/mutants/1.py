import numpy as np

def confusion_matrix_counts(truth,predicted,classes):
    y=np.asarray(truth,dtype=int);p=np.asarray(predicted,dtype=int);out=np.zeros((classes,classes),dtype=np.int64)
    np.add.at(out,(p,y),1);return out
