from abc import abstractmethod
from typing import Tuple, Callable, Dict

import torch
import numpy as np
from e2cnn.nn import GeometricTensor, FieldType, Grid, HybridTensor
from .r2layer import R2Layer
from .basisexpansion_blocks_sparse import SparseBlocksBasisExpansion
from .utils import coo_permute, coo_reshape
import torch_sparse

class R2General(R2Layer):
    def __init__(self,
               in_type: FieldType,
               out_type: FieldType,
               num_neighbors: int,
               in_grid: Grid,
               out_grid: Grid = None,
               **kwargs
               ):
        if "groups" in kwargs and kwargs["groups"] != 1:
            raise NotImplementedError("Groups are currently not implemented for general grids.")

        assert isinstance(in_grid, Grid)
        self.in_grid = in_grid

        if out_grid is None:
            out_grid = in_grid
        assert isinstance(out_grid, Grid)
        self.out_grid = out_grid

        self.num_neighbors = num_neighbors

        super().__init__(in_type, out_type, **kwargs)

    @abstractmethod
    def _compute_basis_params(self, custom_basis_filter, **kwargs) -> Tuple[Callable[[Dict], bool], Dict]:
        """This is the function that needs to be overwritten for a fully specified layer.
        It should return the basis_filter and a dict of params to pass to the
        basisexpansion module."""
        pass

    def forward(self, input):
        # we override the forward implementation, instead of using
        # _forward. The reason is that we want to allocate the filter and bias here,
        # so that we can immediately release the filter after converting
        # it to a sparse tensor, saving a bit of memory
        assert input.type == self.in_type
        assert input.grid == self.in_grid
        filter, bias = self.expand_parameters()

        # Now we implement the sparse convolution.
        # Again, we do this directly here to release memory where possible
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
        input = input.tensor.view(b, c_in * n_in).T
        out = torch_sparse.spmm(indices, values, shape[0], shape[1], input)
        # now we have a dense (c_out * n_out, b) tensor
        # finally, we reshape to (b, c_out, n_out)
        out = out.T.reshape(b, c_out, n_out)

        if bias is not None:
            out = out + bias[:, None]

        return GeometricTensor(out, self.out_type, self.out_grid)
    
    def _forward(self, input: GeometricTensor, filter, bias):
        # this shouldn't be called, since we override self.forward() already
        raise Exception("Something has gone really wrong")

    def _init_basisexpansion(self, in_type, out_type, recompute, basis_filter, basisexpansion, maximum_offset, **kwargs):
        basis_filter, params = self._compute_basis_params(custom_basis_filter=basis_filter, **kwargs)

        # BasisExpansion: submodule which takes care of building the filter
        self._basisexpansion = None

        # notice that `in_type` is used instead of `self.in_type` such that it works also when `groups > 1`
        if basisexpansion == 'blocks':
            self._basisexpansion = SparseBlocksBasisExpansion(in_type, out_type,
                                                              self.in_grid, self.out_grid,
                                                              num_neighbors=self.num_neighbors,
                                                              maximum_offset=maximum_offset,
                                                              basis_filter=basis_filter,
                                                              recompute=recompute,
                                                              **params)

        else:
            raise ValueError('Basis Expansion algorithm "%s" not recognized' % basisexpansion)
        shape = (out_type.size * len(self.out_grid), in_type.size * len(self.in_grid))

    def extra_repr(self):
        s = ('{in_type}, {out_type}, num_neighbors={num_neighbors}')
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def expand_parameters(self):
        _filter = self.basisexpansion(self.weights)
        _bias = self._expand_bias()

        return _filter, _bias

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        raise NotImplementedError()

    def _apply(self, fn):
        super()._apply(fn)
        # HACK: It's tricky to find out whether fn was a function
        # that moved the module to a different device.
        # But we can just make sure we are on the right device everytime
        # _apply is called:
        if hasattr(self, "filter"):
            self.filter = self.filter.to(self.weights.device)
    
    @property
    def is_irregular(self):
        return True
