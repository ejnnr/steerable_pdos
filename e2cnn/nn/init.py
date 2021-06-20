

import warnings
from e2cnn.nn.modules.r2_conv.basisexpansion import BasisExpansion

from collections import defaultdict

import torch
from scipy import stats
import math

__all__ = ["generalized_he_init", "deltaorthonormal_init"]

def regular_init(tensor: torch.Tensor, basisexpansion: BasisExpansion):
    assert tensor.shape == (basisexpansion.dimension(), )
    in_size = basisexpansion._input_size
    out_size = len(basisexpansion._out_type)

    # this init is tailored to a particular basis, it won't work for others
    assert all(r.name == "regular" or r.name == "irrep_0" for r in basisexpansion._in_type.representations)
    assert all(r.name == "regular" or r.name == "irrep_0" for r in basisexpansion._out_type.representations)
    assert basisexpansion.dimension() == in_size * out_size * 9, f"Expected {in_size * out_size * 9} weights, got {basisexpansion.dimension()}"

    # indices of each first channel
    indices = torch.from_numpy(basisexpansion._out_type.fields_start.astype(int))

    xavier_weights = torch.empty(out_size, in_size, 3, 3)
    torch.nn.init.xavier_normal_(xavier_weights)

    # what we need to solve is just a linear system,
    # but explicitly setting that up is pretty error-prone.
    # Instead, we use gradient descent to find an approximate solution
    weights = torch.randn_like(tensor, requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8)
    for i in range(100):
        optimizer.zero_grad()
        filters = basisexpansion(weights).reshape(basisexpansion._output_size, in_size, 5, 5)
        filters = filters[indices]
        # take only the middle 3x3 part
        filters = filters[:, :, 1:-1, 1:-1]
        loss = ((filters - xavier_weights)**2).mean()
        loss.backward()
        optimizer.step()
        if loss.item() < 1e-4:
            break
        if i % 10 == 0:
            scheduler.step()
    
    if loss.item() > 1e-3:
        warnings.warn(f"Init not converged: final loss is {loss.item()}")
    
    tensor[:] = weights.detach()

def generalized_he_init(tensor: torch.Tensor, basisexpansion: BasisExpansion):
    r"""
    
    Initialize the weights of a convolutional layer with a generalized He's weight initialization method.
    
    Args:
        tensor (torch.Tensor): the tensor containing the weights
        basisexpansion (BasisExpansion): the basis expansion method

    """
    # Initialization
    
    assert tensor.shape == (basisexpansion.dimension(), )
    
    vars = torch.ones_like(tensor)
    
    inputs_count = defaultdict(lambda: set())
    basis_count = defaultdict(int)
    
    for attr in basisexpansion.get_basis_info():
        i, o = attr["in_irreps_position"], attr["out_irreps_position"]
        in_irrep, out_irrep = attr["in_irrep"], attr["out_irrep"]
        inputs_count[o].add(in_irrep)
        basis_count[(in_irrep, o)] += 1
    
    for o in inputs_count.keys():
        inputs_count[o] = len(inputs_count[o])
    
    for w, attr in enumerate(basisexpansion.get_basis_info()):
        i, o = attr["in_irreps_position"], attr["out_irreps_position"]
        in_irrep, out_irrep = attr["in_irrep"], attr["out_irrep"]
        vars[w] = 1. / math.sqrt(inputs_count[o] * basis_count[(in_irrep, o)])
    
    # for i, o in basis_count.keys():
    #     print(i, o, inputs_count[o],  basis_count[(i, o)])
    
    tensor[:] = vars * torch.randn_like(tensor)


def deltaorthonormal_init(tensor: torch.Tensor, basisexpansion: BasisExpansion):
    r"""
    
    Initialize the weights of a convolutional layer with *delta-orthogonal* initialization.
    
    Args:
        tensor (torch.Tensor): the tensor containing the weights
        basisexpansion (BasisExpansion): the basis expansion method

    """
    # Initialization

    assert tensor.shape == (basisexpansion.dimension(), )
    
    tensor.fill_(0.)
    
    counts = defaultdict(lambda: defaultdict(lambda: []))
    
    for p, attr in enumerate(basisexpansion.get_basis_info()):
        i = attr["in_irrep"]
        o = attr["out_irrep"]
        ip = attr["in_irreps_position"]
        op = attr["out_irreps_position"]
        if "radius" in attr:
            r = attr["radius"]
        elif "order" in attr:
            r = attr["order"]
        else:
            raise ValueError("Attribute dict has no radial information, needed for deltaorthonormal init")
        
        if i == o and r == 0.:
            counts[i][(ip, op)].append(p)
    
    def same_content(l):
        l = list(l)
        return all(ll == l[0] for ll in l)
    
    for irrep, count in counts.items():
        assert same_content([len(x) for x in count.values()]), [len(x) for x in count.values()]
        in_c = defaultdict(int)
        out_c = defaultdict(int)
        for ip, op in count.keys():
            in_c[ip] += 1
            out_c[op] += 1
        assert same_content(in_c.values()), count.keys()
        assert same_content(out_c.values()), count.keys()
        
        assert list(in_c.values())[0] == len(out_c.keys())
        assert list(out_c.values())[0] == len(in_c.keys())
        
        s = len(list(count.values())[0])
        i = len(in_c.keys())
        o = len(out_c.keys())
        
        # assert i <= o, (i, o, s, irrep, self._input_size, self._output_size)
        # if i > o:
        #     print("Warning: using delta orthogonal initialization to map to a larger number of channels")
        
        if max(o, i) > 1:
            W = stats.ortho_group.rvs(max(i, o))[:o, :i]
            # W = np.eye(o, i)
        else:
            W = 2 * torch.randint(0, 1, size=(1, 1)) - 1
            # W = np.array([[1]])
        
        w = torch.randn((o, s))
        # w = torch.ones((o, s))
        w *= 5.
        w /= (w ** 2).sum(dim=1, keepdim=True).sqrt()
        
        for i, ip in enumerate(in_c.keys()):
            for o, op in enumerate(out_c.keys()):
                for p, pp in enumerate(count[(ip, op)]):
                    tensor[pp] = w[o, p] * W[o, i]

