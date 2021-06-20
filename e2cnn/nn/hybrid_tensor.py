from typing import Tuple

import torch

class HybridTensor:

    def __init__(self, indices: torch.Tensor, values: torch.Tensor, sparse_length: int):
        """A tensor that is sparse in the last dimension, with a fixed sparsity pattern
        over the last two.

        Note that duplicate indices are not allowed, in contrast to sparse Pytorch tensors!
        
        Args:
            indices: (n_out, num_neighbors) tensor for the nearest neighbor indices
            values: (..., n_out, num_neighbors) tensor with the values
            sparse_length: the length of the sparse (final) dimension"""
        assert len(values.shape) >= 2
        assert sparse_length >= values.shape[-1]
        assert indices.shape == values.shape[-2:]
        self.indices = indices
        self.values = values
        self.shape = values.shape[:-1] + (sparse_length, )
        self._batch_shape = values.shape[:-2]
        self._spatial_shape = indices.shape
        self._full_spatial_shape = self.shape[-2:]

    def left_multiply(self, W: torch.Tensor) -> "HybridTensor":
        """Calculate W @ self, where W is a dense tensor. The summation
        is done over the last axis of W and the first axis of self.
        
        Returns:
            A new HybridTensor"""
        values = torch.matmul(W, self.values.moveaxis(0, -2)).moveaxis(-2, 0)
        return HybridTensor(self.indices, values, self.shape[-1])
    
    def view(self, *args) -> "HybridTensor":
        """Create a view of the batch part of the HybridTensor.

        Args:
            shape: a tuple for the new shape of the non-spatial dimensions (... in __init__)"""
        if len(args) == 1:
            assert isinstance(args[0], Tuple)
            shape = args[0]
        else:
            shape = args
        return HybridTensor(self.indices, self.values.view(shape + self._spatial_shape), self.shape[-1])

    def reshape(self, *args) -> "HybridTensor":
        """Reshape the batch part of the HybridTensor.

        Args:
            shape: a tuple for the new shape of the non-spatial dimensions (... in __init__)"""
        if len(args) == 1:
            assert isinstance(args[0], Tuple)
            shape = args[0]
        else:
            shape = args
        return HybridTensor(self.indices, self.values.reshape(shape + self._spatial_shape), self.shape[-1])
    
    def permute(self, *args) -> "HybridTensor":
        """Permute the batch part of the HybridTensor.

        Args:
            permutation: a tuple with a permutation of the batch axes."""
        if len(args) == 1:
            assert isinstance(args[0], Tuple)
            permutation = args[0]
        else:
            permutation = args

        l = len(self.shape)
        return HybridTensor(
            self.indices,
            # leave the last two axes as they are:
            self.values.permute(permutation + (l - 2, l - 1)),
            self.shape[-1]
        )
    
    def filter(self, mask: torch.Tensor) -> "HybridTensor":
        """Create a filtered HybridTensor, with a mask applied along the first axis.

        Args:
            mask (tensor): dense boolean tensor of shape (N, ), where N is the length
            of the first axis of this HybridTensor
        
        Returns:
            HybridTensor of shape (N', ...), where N' is the number of true elements
            in ``mask`` and ... are all but the first axes of this HybridTensor
        """
        assert len(self.shape) >= 3, "Can only filter batch dimensions"
        assert mask.shape == (self.shape[0], )
        return HybridTensor(self.indices, self.values[mask, ...], self.shape[-1])

    @classmethod
    def from_dense(cls, x: torch.Tensor) -> "HybridTensor":
        assert not x.is_sparse
        assert len(x.shape) >= 2
        n_in = x.shape[-1]
        n_out = x.shape[-2]
        indices = torch.arange(n_in).repeat(n_out, 1)
        return HybridTensor(indices, x, n_in)
    
    def to_dense(self) -> torch.Tensor:
        out = torch.zeros(self.shape)
        n_out = self.shape[-2]
        out[..., torch.arange(n_out)[:, None], self.indices] = self.values
        return out
    
    def to_sparse(self) -> torch.Tensor:
        indices, values = self.coo_data()
        return torch.sparse_coo_tensor(indices, values, self.shape)
    
    def coo_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # We want to create the index tensor with shape (len(self.shape), nnz).
        # First, flatten the indices, this gives us the indices into the last
        # dimension:
        indices = self.indices.view(1, -1)
        # Then we create the indices into the n_out dimension (i.e. the dense
        # spatial one). These indices are just aranges, but with each element
        # repeated
        new_indices = torch.arange(
            self.shape[-2],
            device=indices.device,
            dtype=indices.dtype,
        ).repeat_interleave(self.indices.shape[-1]).view(1, -1)
        # then we stack those together:
        indices = torch.cat((new_indices, indices), dim=0)

        # Now for each batch dimension, we do the same, but additionally
        # repeat the existing indices
        for size in reversed(self._batch_shape):
            n = indices.shape[-1]
            new_indices = torch.arange(
                size, device=indices.device, dtype=indices.dtype
            ).repeat_interleave(n).view(1, -1)
            indices = indices.repeat((1, size))
            indices = torch.cat((new_indices, indices), dim=0)
        
        return indices, self.values.view(-1)
    
    def to(self, device):
        return HybridTensor(self.indices.to(device), self.values.to(device), self.shape[-1])

    def cuda(self):
        return HybridTensor(self.indices.cuda(), self.values.cuda(), self.shape[-1])

    def cpu(self):
        return HybridTensor(self.indices.cpu(), self.values.cpu(), self.shape[-1])