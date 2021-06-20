import itertools

import numpy as np
import sparse
import torch

from e2cnn.diffops.utils import make_grid
from e2cnn.nn.modules.r2_conv.utils import get_grid_coords
from e2cnn.nn.modules.r2_conv import SparseBlocksBasisExpansion
from e2cnn.nn.modules.r2_conv.r2general import sparse_conv
from e2cnn.nn.modules.r2_conv.sparse_basisexpansion_blocks import create_filter, normalize_basis, pad_basis
from e2cnn.nn.modules.r2_conv.basisexpansion_singleblock import normalize_basis as dense_normalize_basis
from e2cnn.nn import R2Conv, R2GeneralConv, GeometricTensor, FieldType, init, Grid
from e2cnn.gspaces import *

def test_basis_creation():
    in_grid = Grid.regular((3, 3))
    out_grid = Grid.regular((3, 3))

    N = 3
    g = Rot2dOnR2(N)

    in_type = FieldType(g, [g.regular_repr])
    out_type = FieldType(g, [g.irrep(1)])

    c_in, c_out = in_type.size, out_type.size
    n_in, n_out = len(in_grid), len(out_grid)

    expansion = SparseBlocksBasisExpansion(
        in_type,
        out_type,
        in_grid,
        out_grid,
        method="kernel",
        num_neighbors=5,
        maximum_offset=0,
        sigma=[0.6, 0.6],
        rings=[0.5, 1]
    )

    assert expansion.basis_tensors.shape == (expansion.dimension(), c_out * n_out, c_in * n_in)

    tensors = expansion.basis_tensors.to_dense().reshape(expansion.dimension(), c_out, n_out, c_in, n_in)
    nonzero = (tensors != 0).float()
    # each output point should have at most num_neighbors nonzero coefficients
    assert torch.all(nonzero.sum(-1) <= 5)

    for attr in expansion.get_basis_info():
        i_start = int(in_type.fields_start[attr["in_field_position"]])
        i_end = int(in_type.fields_end[attr["in_field_position"]])
        o_start = int(out_type.fields_start[attr["out_field_position"]])
        o_end = int(out_type.fields_end[attr["out_field_position"]])
        idx = attr["idx"]
        mask = torch.ones(c_out, n_out, c_in, n_in)
        mask[o_start:o_end, :, i_start:i_end, :] = 0
        mask = mask.reshape(c_out * n_out, c_in * n_in)
        mask = mask.to_sparse()
        # the basis tensor should be zero outside its block
        assert torch.all((expansion.basis_tensors[idx] * mask).values() == 0)

def test_basis_expansion():
    in_grid = Grid.regular((3, 3))
    out_grid = Grid.regular((3, 3))

    N = 3
    g = Rot2dOnR2(N)

    in_type = FieldType(g, [g.regular_repr])
    out_type = FieldType(g, [g.irrep(1)])

    c_in, c_out = in_type.size, out_type.size
    n_in, n_out = len(in_grid), len(out_grid)

    expansion = SparseBlocksBasisExpansion(
        in_type,
        out_type,
        in_grid,
        out_grid,
        method="kernel",
        num_neighbors=5,
        maximum_offset=0,
        sigma=[0.6, 0.6],
        rings=[0.5, 1]
    )
    tensors = expansion.basis_tensors.to_dense()
    weights = torch.randn(expansion.dimension())
    correct = (tensors * weights[:, None, None]).sum(0)
    result = expansion(weights).to_dense()
    assert torch.allclose(result, correct, atol=1e-5)


def test_module_on_regular_grid_random():
    N = 3
    g = Rot2dOnR2(N)

    r1 = FieldType(g, [g.regular_repr])
    r2 = FieldType(g, [g.irrep(1)])
    # r1 = FieldType(g, [g.trivial_repr])
    # r2 = FieldType(g, [g.regular_repr])
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

    regular_model = R2Conv(r1, r2,
                           kernel_size=3,
                           rings=[0.5, 1],
                           bias=False,
                           init=None)
    # To make sure the results are the same, we use the same weight
    # for each filter. This is necessary because the two models associate
    # the weights to different filters
    regular_model.weights.data = torch.ones_like(regular_model.weights.data)

    assert x_general.shape == (B, regular_model.in_type.size, len(grid))
    general_model = R2GeneralConv(r1, r2,
                                  rings=[0.5, 1],
                                  in_grid=grid,
                                  num_neighbors=9,
                                  bias=False,
                                  init=None)
    general_model.weights.data = torch.ones_like(regular_model.weights.data)

    assert general_model.basisexpansion.dimension() == regular_model.basisexpansion.dimension()

    regular_out = regular_model(x).tensor
    out = general_model(x_general).tensor.reshape(B, regular_model.out_type.size, P, P)
    # discard the border pixels (which aren't present for convolutions)
    out = out[:, :, 1:-1, 1:-1]

    if not torch.allclose(out, regular_out, atol=1e-5):
        print(np.around(out[0, 0].detach().numpy(), 2))
        print(np.around(regular_out[0, 0].detach().numpy(), 2))
    assert torch.allclose(out, regular_out, atol=1e-5)

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

    regular_model = R2Conv(r1, r2,
                           kernel_size=3,
                           rings=[0.5, 1],
                           bias=False,
                           init=None)
    # To make sure the results are the same, we use the same weight
    # for each filter. This is necessary because the two models associate
    # the weights to different filters
    regular_model.weights.data = torch.ones_like(regular_model.weights.data)

    assert x_general.shape == (B, regular_model.in_type.size, len(grid))
    general_model = R2GeneralConv(r1, r2,
                                  rings=[0.5, 1],
                                  in_grid=grid,
                                  num_neighbors=9,
                                  bias=False,
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
