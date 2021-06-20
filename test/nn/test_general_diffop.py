import itertools

import numpy as np
import torch_sparse
import torch

from e2cnn.diffops.utils import make_grid
from e2cnn.nn.modules.r2_conv.utils import get_grid_coords, coo_permute, coo_reshape
from e2cnn.nn.modules.r2_conv import SparseBlocksBasisExpansion
from e2cnn.nn.modules.r2_conv.basisexpansion_singleblock_sparse import SparseSingleBlockBasisExpansion, normalize_basis
from e2cnn.nn.modules.r2_conv.basisexpansion_singleblock import normalize_basis as dense_normalize_basis
from e2cnn.nn import R2Diffop, R2GeneralDiffop, GeometricTensor, FieldType, init, Grid, HybridTensor
from e2cnn.gspaces import *

# HACK: this function isn't actually used anywhere, but it's content
# is the main part of R2General._forward.
# It's inlined there so we can immediately release some variables
# and free up memory. It's reproduced here only for testing purposes
def sparse_conv(filter: HybridTensor, input: torch.Tensor) -> torch.Tensor:
    """Compute a convolution of a dense input with a sparse filter.

    Args:
        filter (HybridTensor): must have shape (c_out, c_in, n_out, n_in)
        input (dense tensor): shape (b, c_in, n_in) (where b is the batch dimension)

    Returns:
        dense tensor of shape (b, c_out, n_out)"""
    b, c_in, n_in = input.shape
    c_out, n_out = filter.shape[0], filter.shape[2]
    # now we turn the filter into a sparse matrix, and then multiply
    # along the spatial input axis.
    # Converting to a sparse tensor isn't ideal but I think it's the best
    # we can do if we want to use Pytorch's matrix multiplication.

    # This helps a bit to avoid GPU fragmentation.
    # TODO: there is no deep reason to put this here, maybe somewhere else
    # would be even better?
    torch.cuda.empty_cache()
    indices, values = filter.coo_data()
    shape = filter.shape
    # We don't need the filter in its HybridTensor form anymore
    del filter
    # reshape filter to (c_out * n_out, c_in * n_in)
    indices, values, shape = coo_permute(indices, values, shape, (0, 2, 1, 3))
    indices, values, shape = coo_reshape(indices, values, shape, (c_out * n_out, c_in * n_in))
    # reshape input to (c_in * n_in, b)
    input = input.view(b, c_in * n_in).T
    out = torch_sparse.spmm(indices, values, shape[0], shape[1], input)
    # now we have a dense (c_out * n_out, b) tensor
    # finally, we reshape to (b, c_out, n_out)
    return out.T.reshape(b, c_out, n_out)

def test_sparse_conv():
    c_out, c_in, n_out, n_in = (2, 3, 5, 4)
    batch_size = 10
    filter = torch.randn(c_out, c_in, n_out, n_in)
    hybrid_filter = HybridTensor.from_dense(filter)
    input = torch.randn(batch_size, c_in, n_in)

    conv_out = sparse_conv(hybrid_filter, input)
    correct = torch.einsum("oipq,biq->bop", filter, input)

    # There can sometimes be numerical errors due to floating point precision,
    # so we increase the tolerance a bit
    assert torch.allclose(conv_out, correct, atol=1e-5)


def test_normalize_basis():
    # the n_out dimension should be 1, otherwise we can't compare the result
    # to the dense version (which only supports a single output point)
    b = torch.randn(4, 3, 2, 1, 3)
    sizes = torch.tensor([2] * 4)
    dense_basis = dense_normalize_basis(b, sizes)
    sparse_basis = normalize_basis(HybridTensor.from_dense(b), sizes).to_dense()

    assert torch.allclose(sparse_basis, dense_basis, atol=1e-5)


def test_basis_expansion():
    in_grid = Grid.regular((3, 3))
    out_grid = Grid.regular((3, 3))

    N = 3
    g = Rot2dOnR2(N)

    basis = g.build_kernel_basis(g.regular_repr, g.irrep(1),
                                 method="diffop",
                                 max_power=1,
                                 maximum_frequency=1,
                                 )

    expansion = SparseSingleBlockBasisExpansion(
        basis,
        in_grid,
        out_grid,
        num_neighbors=5,
        basis_filter=lambda x: True,
    )
    tensors = expansion.sampled_basis.to_dense()
    weights = torch.randn(5, expansion.dimension())
    correct = (tensors[None] * weights[:, :, None, None, None, None]).sum(1)
    result = expansion(weights).to_dense()
    assert torch.allclose(result, correct, atol=1e-5)


def test_module_on_regular_grid_random():
    N = 3
    g = Rot2dOnR2(N)

    r1 = FieldType(g, [g.regular_repr])
    r2 = FieldType(g, [g.irrep(1)])
    P = 7
    B = 1
    x = torch.rand(B, r1.size, P, P)
    x_general = x.reshape(B, r1.size, P * P)
    x = GeometricTensor(x, r1)
    # create 7x7 grid
    # for now we need to use this function instead of Grid.regular
    # because otherwise it's inconsistent with the convolutions used by R2Diffop
    grid = Grid(get_grid_coords(7))
    x_general = GeometricTensor(x_general, r1, grid)

    regular_model = R2Diffop(r1, r2,
                             kernel_size=3,
                             maximum_order=2,
                             cache=False,
                             rbffd=True,
                             bias=False)

    assert x_general.shape == (B, regular_model.in_type.size, len(grid))
    general_model = R2GeneralDiffop(r1, r2,
                                    maximum_order=2,
                                    cache=False,
                                    in_grid=grid,
                                    num_neighbors=9,
                                    bias=False,
                                    init=None)
    # use the same weights as the regular model
    general_model.weights.data = regular_model.weights.data

    assert general_model.basisexpansion.dimension() == regular_model.basisexpansion.dimension()

    regular_out = regular_model(x).tensor
    out = general_model(x_general).tensor.reshape(B, regular_model.out_type.size, P, P)
    # discard the border pixels (which aren't present for convolutions)
    out = out[:, :, 1:-1, 1:-1]

    if not torch.allclose(out, regular_out, atol=1e-5):
        print(np.around(out[0, 0].detach().numpy(), 2))
        print(np.around(regular_out[0, 0].detach().numpy(), 2))
    assert torch.allclose(out, regular_out, atol=1e-5)

def test_backward_pass():
    N = 3
    g = Rot2dOnR2(N)

    r1 = FieldType(g, [g.regular_repr])
    r2 = FieldType(g, [g.irrep(1)])
    P = 7
    B = 1
    x = torch.rand(B, r1.size, P, P)
    x_general = x.reshape(B, r1.size, P * P)
    x = GeometricTensor(x, r1)
    # create 7x7 grid
    # for now we need to use this function instead of Grid.regular
    # because otherwise it's inconsistent with the convolutions used by R2Diffop
    grid = Grid(get_grid_coords(7))
    x_general = GeometricTensor(x_general, r1, grid)

    regular_model = R2Diffop(r1, r2,
                             kernel_size=3,
                             maximum_order=2,
                             cache=False,
                             rbffd=True,
                             bias=False)

    assert x_general.shape == (B, regular_model.in_type.size, len(grid))
    general_model = R2GeneralDiffop(r1, r2,
                                    maximum_order=2,
                                    cache=False,
                                    in_grid=grid,
                                    num_neighbors=9,
                                    bias=False)
    # use the same weights as the regular model
    # the .detach() is so that the gradients don't interfere
    general_model.weights.data.copy_(regular_model.weights.data)

    assert general_model.basisexpansion.dimension() == regular_model.basisexpansion.dimension()

    regular_out = regular_model(x).tensor
    out = general_model(x_general).tensor.reshape(B, regular_model.out_type.size, P, P)
    # discard the border pixels (which aren't present for convolutions)
    out = out[:, :, 1:-1, 1:-1]

    assert torch.allclose(out, regular_out, atol=1e-5)

    # our "loss" function will be a weighted sum of the output
    # the random weights ensure that if there are any differences
    # in the backward pass, they'll almost certainly show up
    coefficients = torch.randn_like(out)
    regular_loss = (regular_out * coefficients).sum()
    loss = (out * coefficients).sum()
    regular_loss.backward()
    loss.backward()

    assert torch.allclose(general_model.weights.grad, regular_model.weights.grad, atol=1e-5)

def test_module_on_regular_grid_simple():
    N = -1
    g = Rot2dOnR2(N, maximum_frequency=2)

    r1 = FieldType(g, [g.irrep(1)])
    r2 = FieldType(g, [g.irrep(1)])
    # r1 = FieldType(g, [g.trivial_repr])
    # r2 = FieldType(g, [g.regular_repr])
    P = 7
    B = 1
    x = torch.zeros(B, r1.size, P, P)
    # insert a peak in the middle of the input
    x[:, :, 3, 3] = 1
    x_general = x.reshape(B, r1.size, P * P)
    x = GeometricTensor(x, r1)
    # create 7x7 grid
    grid = Grid(get_grid_coords(7))
    x_general = GeometricTensor(x_general, r1, grid)

    regular_model = R2Diffop(r1, r2,
                             kernel_size=3,
                             maximum_order=2,
                             cache=False,
                             rbffd=True,
                             init=None)
    # To make sure the results are the same, we use the same weight
    # for each filter. This is necessary because the two models associate
    # the weights to different filters
    regular_model.weights.data = torch.ones_like(regular_model.weights.data)

    assert x_general.shape == (B, regular_model.in_type.size, len(grid))
    general_model = R2GeneralDiffop(r1, r2,
                                    maximum_order=2,
                                    cache=False,
                                    in_grid=grid,
                                    num_neighbors=9,
                                    init=None)
    general_model.weights.data = torch.ones_like(regular_model.weights.data)

    assert general_model.basisexpansion.dimension() == regular_model.basisexpansion.dimension()

    regular_out = regular_model(x).tensor
    out = general_model(x_general).tensor.reshape(B, regular_model.out_type.size, P, P)
    # discard the border pixels (which aren't present for convolutions)
    out = out[:, :, 1:-1, 1:-1]

    if not torch.allclose(out, regular_out, atol=1e-5):
        print(out.shape, regular_out.shape)
        print(general_model.basisexpansion.basis_tensors.shape)
        print(np.around(out[0].detach().numpy(), 2))
        print(np.around(regular_out[0].detach().numpy(), 2))
    assert torch.allclose(out, regular_out, atol=1e-5)
