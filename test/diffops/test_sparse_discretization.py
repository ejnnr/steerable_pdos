import numpy as np
from e2cnn.diffops.utils import *
import sparse

def test_output_format():
    in_coords = make_grid(2)
    diffop = np.array([1, 1, 1])
    # shift the grid slightly and remove a few points
    out_coords = in_coords[:, :20] + 0.5
    num_neighbors = 9
    out = discretize_homogeneous_polynomial(in_coords, diffop, out_coords, num_neighbors)
    assert isinstance(out, sparse.COO)
    assert out.shape == (out_coords.shape[1], in_coords.shape[1])
    for i in range(out_coords.shape[1]):
        # the number of nonzero values might be less than the number of neighbors
        # if some entries happen to be zero
        assert out[i].nnz <= num_neighbors
        # but there should always be at least one nonzero weight for each output point
        assert out[i].nnz > 0
