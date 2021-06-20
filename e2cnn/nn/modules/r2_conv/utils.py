from typing import Tuple

import numpy as np
import torch


def get_grid_coords(kernel_size: int, dilation: int = 1):

    actual_size = dilation * (kernel_size -1) + 1

    origin = actual_size / 2 - 0.5

    points = []

    for y in range(kernel_size):
        y *= dilation
        for x in range(kernel_size):
            x *= dilation
            p = (x - origin, -y + origin)
            points.append(p)

    points = np.array(points)
    assert points.shape == (kernel_size ** 2, 2), points.shape
    return points.T


def coo_reshape(
    indices: torch.Tensor,
    values: torch.Tensor,
    initial_shape: Tuple[int, ...],
    target_shape: Tuple[int, ...]
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    indices = ravel_indices(indices, initial_shape)
    indices = unravel_indices_(indices, target_shape)
    return indices, values, target_shape


def coo_permute(
    indices: torch.Tensor,
    values: torch.Tensor,
    shape: Tuple[int, ...],
    permutation: Tuple[int, ...],
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    indices = indices[permutation, :]
    shape = tuple(shape[i] for i in permutation)
    return indices, values, shape


def sparse_reshape(x: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    # we use ._indices() instead of .indices() so that this function
    # also works un uncoalesced tensors. Coalescing is too memory intensive
    indices = x._indices()
    indices = ravel_indices(indices, x.shape)
    indices = unravel_indices_(indices, shape)
    # TODO: tensor._values() doesn't include gradient_fn,
    # which means that currently this function doesn't support autograd.
    # OTOH, coalescing the huge filters (so that we could use .values())
    # leads to out of memory issues
    return sparse_tensor_unsafe(indices, x._values(), shape)


def sparse_permute(x: torch.Tensor, permutation: Tuple[int, ...]) -> torch.Tensor:
    # we use ._indices() instead of .indices() so that this function
    # also works un uncoalesced tensors. Coalescing is too memory intensive
    indices = x._indices()[permutation, :]
    shape = tuple(x.shape[i] for i in permutation)
    # TODO: tensor._values() doesn't include gradient_fn,
    # which means that currently this function doesn't support autograd.
    # OTOH, coalescing the huge filters (so that we could use .values())
    # leads to out of memory issues
    return sparse_tensor_unsafe(indices, x._values(), shape)


def ravel_indices(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of indices, (D, N).
        shape: The input shape, (D, )

    Returns:
        The raveled coordinates, (*, N).
    """
    # how much to multiply the indices in each axis by
    multipliers = torch.cumprod(torch.tensor(
        (1, ) + shape[:0:-1],
        dtype=indices.dtype,
        device=indices.device,
    ), dim=0)
    # It's important to flip the multipliers here and not the indices!
    # torch.flip makes a copy (there is no other way to flip in pytorch),
    # and indices can be multiple GB
    multipliers = torch.flip(multipliers, (0, ))
    return (indices * multipliers[:, None]).sum(dim=0)

# See https://github.com/pytorch/pytorch/issues/35674
def unravel_indices_(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Warning: this modifies the input tensor to save memory! The indices tensor
    will be nonsense after applying this function, so make a copy if you need one!

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, D, N).
    """

    batch_shape = indices.shape[:-1]
    D = len(shape)
    N = indices.shape[-1]
    # Adapted this to allocate the result immediately and
    # fill it in, instead of collecting the rows and stacking them.
    # This saves a lot of memory if indices is a huge tensor
    coords = torch.empty(batch_shape + (D, N), dtype=indices.dtype, device=indices.device)

    for i, dim in enumerate(reversed(shape)):
        coords[..., D - i - 1, :] = indices % dim
        # we modify the indices in-place to avoid making a copy
        indices //= dim

    return coords


def sparse_tensor_unsafe(index, data, size, coalesced=False):
    # helper function to create a sparse pytorch tensor without checking
    # that the indices are in bounds.
    # See https://github.com/pytorch/pytorch/issues/14945
    t = torch._sparse_coo_tensor_unsafe(index, data, size=size, device=index.device, dtype=data.dtype)
    t._coalesced_(coalesced)
    return t


def prepend_axis(indices, length):
    d, n = indices.shape
    new_indices = torch.arange(n, device=indices.device, dtype=indices.dtype).repeat(length).view(1, -1)
    indices = indices.repeat((1, length))
    assert indices.shape == (d, length * n)
    assert new_indices.shape == (1, length * n)
    return torch.cat((new_indices, indices), dim=0)
