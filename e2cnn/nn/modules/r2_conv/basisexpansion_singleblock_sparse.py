import warnings

from e2cnn.kernels import KernelBasis, EmptyBasisException
from e2cnn.diffops import DiffopBasis
from e2cnn.nn import Grid, HybridTensor
from .basisexpansion import BasisExpansion

from typing import Callable, Dict, List, Iterable, Union, Tuple

import torch
import numpy as np
import sparse
from rbf.utils import KDTree

__all__ = ["SparseSingleBlockBasisExpansion", "sparse_block_basisexpansion"]

warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated")


class SparseSingleBlockBasisExpansion(BasisExpansion):
    
    def __init__(self,
                 basis: KernelBasis,
                 in_grid: Grid,
                 out_grid: Grid,
                 num_neighbors: int,
                 basis_filter: Callable[[dict], bool] = None,
                 normalize: bool = True,
                 # ignored for now
                 smoothing: float = None,
                 ):

        super().__init__()

        if smoothing is not None:
            raise NotImplementedError("Gaussians not yet implemented for irregular grids.")
        
        # compute the mask of the sampled basis containing only the elements allowed by the filter
        mask = np.zeros(len(basis), dtype=bool)
        for b, attr in enumerate(basis):
            mask[b] = basis_filter(attr)
            
        if not any(mask):
            raise EmptyBasisException
        
        self.attributes, sampled_basis = sample_basis(basis, in_grid, out_grid, num_neighbors, mask, normalize)
        
        # register the bases tensors as parameters of this module
        # HACK: This should ideally be a buffer, so it would be moved to the right
        # device automatically
        self.sampled_basis = sampled_basis
            
        self._idx_to_ids = []
        self._ids_to_idx = {}
        for idx, attr in enumerate(self.attributes):
            if "radius" in attr:
                radial_info = attr["radius"]
            elif "order" in attr:
                radial_info = attr["order"]
            else:
                raise ValueError("No radial information found.")

            if "in_irrep" in attr:
                id = '({}-{},{}-{})_({}/{})_{}'.format(
                        attr["in_irrep"], attr["in_irrep_idx"],  # name and index within the field of the input irrep
                        attr["out_irrep"], attr["out_irrep_idx"],  # name and index within the field of the output irrep
                        radial_info,
                        attr["frequency"],  # frequency of the basis element
                        # int(np.abs(attr["frequency"])),  # absolute frequency of the basis element
                        attr["inner_idx"],
                        # index of the basis element within the basis of radially independent kernels between the irreps
                    )
            else:
                id = 'special_regular_({}/{})_{}'.format(
                        radial_info,
                        attr["frequency"],  # frequency of the basis element
                        # int(np.abs(attr["frequency"])),  # absolute frequency of the basis element
                        attr["inner_idx"],
                        # index of the basis element within the basis of radially independent kernels between the irreps
                    )
            attr["id"] = id
            self._ids_to_idx[id] = idx
            self._idx_to_ids.append(id)

        # HACK: we want to be able to find out what device this module
        # should be on
        # Inspired by https://stackoverflow.com/a/63477353
        self.register_buffer("device_test_buffer", torch.empty(0), persistent=False)
    
    def forward(self, weights: torch.Tensor) -> HybridTensor:
        # weights must have shape (_, basis length),
        # where _ is treated as a batch dimension, i.e. we effectively
        # broadcast over that one
        assert len(weights.shape) == 2
        assert weights.shape[1] == self.dimension()

        return self.sampled_basis.left_multiply(weights)

    def get_basis_names(self) -> List[str]:
        return self._idx_to_ids

    def get_element_info(self, name: Union[str, int]) -> Dict:
        if isinstance(name, str):
            name = self._ids_to_idx[name]
        return self.attributes[name]

    def get_basis_info(self) -> Iterable:
        return iter(self.attributes)

    def dimension(self) -> int:
        return self.sampled_basis.shape[0]

    # HACK: buffers would be ideal instead, but they only support Tensors.
    # Can we make HybridTensors subclass Tensors maybe?
    def to(self, device):
        super().to(device)
        self.sampled_basis = self.sampled_basis.to(device)
        return self

    def cuda(self):
        super().cuda()
        self.sampled_basis = self.sampled_basis.cuda()
        return self

    def cpu(self):
        super().cpu()
        self.sampled_basis = self.sampled_basis.cpu()
        return self
    
    def _apply(self, fn):
        super()._apply(fn)
        # HACK: It's tricky to find out whether fn was a function
        # that moved the module to a different device.
        # But we can just make sure we are on the right device everytime
        # _apply is called:
        self.sampled_basis = self.sampled_basis.to(self.device_test_buffer.device)

# dictionary storing references to already built basis tensors
# when a new filter tensor is built, it is also stored here
# when the same basis is built again (eg. in another layer), the already existing filter tensor is retrieved
_stored_filters = {}


def sparse_block_basisexpansion(
    basis: KernelBasis,
    in_grid: Grid,
    out_grid: Grid,
    num_neighbors: int,
    basis_filter: Callable[[dict], bool] = None,
    recompute: bool = False,
    smoothing: float = None,
    normalize: bool = True,
) -> SparseSingleBlockBasisExpansion:
    r"""


    Args:
        basis (KernelBasis): basis defining the space of kernels
        points (ndarray): points where the analytical basis should be sampled
        basis_filter (callable):
        recompute (bool, optional): whether to recompute new bases or reuse, if possible, already built tensors.

    """
    
    if not recompute:
        # compute the mask of the sampled basis containing only the elements allowed by the filter
        mask = np.zeros(len(basis), dtype=bool)
        for b, attr in enumerate(basis):
            mask[b] = basis_filter(attr)
        
        key = (basis, mask.tobytes(), in_grid, out_grid, num_neighbors, normalize, smoothing)
        if key not in _stored_filters:
            _stored_filters[key] = SparseSingleBlockBasisExpansion(basis, in_grid, out_grid, num_neighbors, basis_filter, smoothing=smoothing, normalize=normalize)
        
        return _stored_filters[key]
    
    else:
        return SparseSingleBlockBasisExpansion(basis, in_grid, out_grid, num_neighbors, basis_filter, smoothing=smoothing, normalize=normalize)


def sample_basis(basis: KernelBasis,
               in_grid: Grid,
               out_grid: Grid,
               num_neighbors: int,
               mask: np.ndarray,
               normalize: bool,
               ) -> Tuple[List[Dict], HybridTensor]:

        attributes = [attr for b, attr in enumerate(basis) if mask[b]]

        # we need to know the real output size of the basis elements (i.e. without the change of basis and the padding)
        # to perform the normalization
        sizes = []
        for attr in attributes:
            sizes.append(attr["shape"][0])

        # sample the basis on the grid
        if isinstance(basis, DiffopBasis):
            # Masking happens inside the sample method because sampling
            # high-order diffops is expensive and we don't want to do that
            # if it's unnecessary
            sampled_basis = basis.sample(
                in_grid.coordinates,
                out_coords=out_grid.coordinates,
                mask=mask,
                num_neighbors=num_neighbors
            )
        else:
            print(basis)
            # kernel bases don't output grids, so instead we use a wrapper
            # function which can handle output grids and returns sparse arrays
            sampled_basis = sample_kernel_basis(
                basis,
                in_grid.coordinates,
                out_grid.coordinates,
                num_neighbors
            )
            # kernel bases don't implement masking, so we do it here
            sampled_basis = sampled_basis.filter(mask)

        if normalize:
            # normalize the basis
            sizes = torch.tensor(sizes)
            sampled_basis = normalize_basis(sampled_basis, sizes)

        # discard the basis which are close to zero everywhere
        norms = (sampled_basis.values ** 2).reshape((sampled_basis.shape[0], -1)).sum(dim=1) > 1e-2
        if not any(norms):
            raise EmptyBasisException
        sampled_basis = sampled_basis.filter(norms)

        attributes = [attr for b, attr in enumerate(attributes) if norms[b]]

        return attributes, sampled_basis


def normalize_basis(basis: HybridTensor, sizes: torch.Tensor) -> HybridTensor:
    r"""

    Normalize the filters in the input tensor.
    The tensor of shape :math:`(B, O, I, ...)` is interpreted as a basis containing ``B`` filters/elements, each with
    ``I`` inputs and ``O`` outputs. The spatial dimensions ``...`` can be anything.

    .. notice ::
        Notice that the method changes the input tensor inplace

    Args:
        basis (torch.Tensor): tensor containing the basis to normalize
        sizes (torch.Tensor): original input size of the basis elements, without the padding and the change of basis

    Returns:
        the normalized basis (the operation is done inplace, so this is just a reference to the input tensor)

    """

    b = basis.shape[0]
    n_out = basis.shape[3]
    assert len(basis.shape) > 2
    assert sizes.shape == (b,)

    # This einsum computes trace(basis @ basis.T), where basis is c_out x c_in,
    # all batched over axis 0 (enumerating the basis elements) and the spatial
    # axes at the end. The diagonal entries of basis @ basis.T are the squared norms of
    # the c_out vectors over c_in, and summing over those gives us the Frobenius
    # norm.
    # basis is a HybridTensor, and basis.values has shape
    # (b, c_out, c_in, n_out, num_neighbors). We don't care about the
    # spatial part here, so we can in fact just use that for the einsum.
    norms = torch.einsum('bop...,bpo...->b...', (basis.values, basis.values.transpose(1, 2)))

    # now norms has shape [b, n_out, num_neighbors].
    # We sum over the num_neighbors axis, to get the norm
    # for each basis element and output point.
    # This is a notable difference from the regular grid case,
    # where we have only one output point and sum over all spatial
    # dimensions.
    norms = norms.sum(dim=-1)
    # Removing the change of basis, these matrices should be multiples of the identity
    # where the scalar on the diagonal is the variance
    # in order to find this variance, we can compute the trace (which is invariant to the change of basis)
    # and divide by the number of elements in the diagonal ignoring the padding.
    # Therefore, we need to know the original size of each basis element.
    norms /= sizes[:, None]

    norms[norms < 1e-15] = 0

    norms = torch.sqrt(norms)

    norms[norms < 1e-6] = 1
    norms[norms != norms] = 1

    # divide by the norm
    basis.values /= norms[:, None, None, :, None]

    return basis

def sample_kernel_basis(basis: KernelBasis,
                        in_coords: np.ndarray,
                        out_coords: np.ndarray,
                        num_neighbors: int) -> sparse.COO:
    # indices will be an ndarray of shape n_out x num_neighbors,
    # where n_out is the number of out points
    _, indices = KDTree(in_coords.T).query(out_coords.T, num_neighbors)
    # We want to query on the difference vectors between input points and
    # output points. We create an array with shape (D, n_out, num_neighbors)
    # with these differences
    query_points = in_coords[:, indices] - out_coords[:, np.arange(len(indices)), None]
    # flatten the last dimensions because that's what KernelBasis.sample expects
    query_points = query_points.reshape(query_points.shape[0], -1)
    sampled_basis = basis.sample(query_points)
    c_out, c_in, b, n_points = sampled_basis.shape
    # now move the basis dimension to the front
    sampled_basis = np.moveaxis(sampled_basis, 2, 0)
    # and unflatten the spatial dimensions
    sampled_basis = sampled_basis.reshape(b, c_out, c_in, out_coords.shape[1], num_neighbors)

    # HACK: Creating a dense array is completely unnecessary and consumes more
    # memory. We should instead calculate the indices into the flattened array ourselves.
    dense_basis = np.zeros((b, c_out, c_in, out_coords.shape[1], in_coords.shape[1]))
    # use NumPy's advanced indexing with broadcasting to set the right elements.
    # indices has shape n_out x n_neighbors, we use np.arange(n_out) for the out dimension
    # and broadcast that to the same shape. Effectively, this indexes the last two
    # dimensions of dense_basis using indices.
    dense_basis[:, :, :, np.arange(out_coords.shape[1])[:, None], indices] = sampled_basis
    return sparse.COO.from_numpy(dense_basis)
