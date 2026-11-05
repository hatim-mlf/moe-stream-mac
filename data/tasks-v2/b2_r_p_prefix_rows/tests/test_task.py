import numpy as np
from candidate import copy_prefix_rows
def test_copy_and_empty():
 a=np.arange(12,dtype=np.int16).reshape(4,3)
 got=copy_prefix_rows(a,2)
 assert got.dtype==np.int16 and np.array_equal(got,a[:2])
 got[0,0]=-1
 assert a[0,0]==0 and copy_prefix_rows(a,0).shape==(0,3)
def test_bounds_and_rank():
 import pytest
 with pytest.raises(ValueError): copy_prefix_rows([[1]],2)
 with pytest.raises(ValueError): copy_prefix_rows([[1]],-1)
 with pytest.raises(ValueError): copy_prefix_rows([[1]],True)
 with pytest.raises(ValueError): copy_prefix_rows([1,2],1)
