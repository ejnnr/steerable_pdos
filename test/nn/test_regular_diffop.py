import numpy as np
import torch

from e2cnn.group import *
from e2cnn.gspaces import *
from e2cnn.nn import *

def check_quarter_rotations(model, elements):
    P = 10
    B = 100
    x = GeometricTensor(torch.rand(B, model.in_type.size, P, P), model.in_type)

    for k, el in enumerate(elements):
        out1 = model(x).transform(el).tensor.detach().numpy()
        out2 = model(x.transform(el)).tensor.detach().numpy()

        # Since we're using Pytorch and thus floating point numbers,
        # we only have a precision of about 1e-6,
        # so we need to increase the tolerance of np.allclose
        assert np.allclose(out1, out2, atol=1e-5), f"element {el}: error of {np.max(np.abs(out1 - out2))}"


def test_cyclic():
    N = 8
    g = Rot2dOnR2(N)

    r1 = FieldType(g, list(g.representations.values()))
    r2 = FieldType(g, list(g.representations.values()) * 2)
    # r1 = FieldType(g, [g.trivial_repr])
    # r2 = FieldType(g, [g.regular_repr])

    cl = R2Diffop(r1, r2,
                  maximum_order=2,
                  cache=False,
                  rbffd=False,
                  init=None)
    init.generalized_he_init(cl.weights.data, cl.basisexpansion)
    cl.bias.data = 20 * torch.randn_like(cl.bias.data)
    check_quarter_rotations(cl, [0, N // 4, N // 2, 3 * (N // 4)])
