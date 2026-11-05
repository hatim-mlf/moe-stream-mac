import numpy as np
from candidate import means_by_label
def test_sorted_labels_and_repeated_values():
 ids,means=means_by_label([10,2,6,4],[7,2,7,2])
 assert np.array_equal(ids,[2,7]) and np.allclose(means,[3,8])
 ids,means=means_by_label([5],[-3])
 assert np.array_equal(ids,[-3]) and np.array_equal(means,[5])
def test_invalid():
 import pytest
 with pytest.raises(ValueError): means_by_label([1,2],[1])
 with pytest.raises(ValueError): means_by_label([np.nan],[1])
 with pytest.raises(ValueError): means_by_label([1],[1.0])
 with pytest.raises(ValueError): means_by_label([],[])
