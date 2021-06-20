
from e2cnn.nn import FieldType, Grid

from .r2regular import R2Regular
from .r2general import R2General
from .utils import get_grid_coords

from typing import Callable, Union, Tuple, List, Optional

import torch
import numpy as np
import math


__all__ = ["R2Conv", "R2GeneralConv"]


class R2Conv(R2Regular):
    
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 kernel_size: int,
                 padding: int = 0,
                 stride: int = 1,
                 dilation: int = 1,
                 padding_mode: str = 'zeros',
                 groups: int = 1,
                 bias: bool = True,
                 basisexpansion: str = 'blocks',
                 sigma: Union[List[float], float] = None,
                 frequencies_cutoff: Union[float, Callable[[float], int]] = None,
                 rings: List[float] = None,
                 maximum_offset: int = None,
                 recompute: bool = False,
                 normalize_basis: bool = True,
                 basis_filter: Callable[[dict], bool] = None,
                 init: Optional[str] = "he"
                 ):
        r"""
        
        
        G-steerable planar convolution mapping between the input and output :class:`~e2cnn.nn.FieldType` s specified by
        the parameters ``in_type`` and ``out_type``.
        This operation is equivariant under the action of :math:`\R^2\rtimes G` where :math:`G` is the
        :attr:`e2cnn.nn.FieldType.fibergroup` of ``in_type`` and ``out_type``.
        
        Specifically, let :math:`\rho_\text{in}: G \to \GL{\R^{c_\text{in}}}` and
        :math:`\rho_\text{out}: G \to \GL{\R^{c_\text{out}}}` be the representations specified by the input and output
        field types.
        Then :class:`~e2cnn.nn.R2Conv` guarantees an equivariant mapping
        
        .. math::
            \kappa \star [\mathcal{T}^\text{in}_{g,u} . f] = \mathcal{T}^\text{out}_{g,u} . [\kappa \star f] \qquad\qquad \forall g \in G, u \in \R^2
            
        where the transformation of the input and output fields are given by
 
        .. math::
            [\mathcal{T}^\text{in}_{g,u} . f](x) &= \rho_\text{in}(g)f(g^{-1} (x - u)) \\
            [\mathcal{T}^\text{out}_{g,u} . f](x) &= \rho_\text{out}(g)f(g^{-1} (x - u)) \\

        The equivariance of G-steerable convolutions is guaranteed by restricting the space of convolution kernels to an
        equivariant subspace.
        As proven in `3D Steerable CNNs <https://arxiv.org/abs/1807.02547>`_, this parametrizes the *most general
        equivariant convolutional map* between the input and output fields.
        For feature fields on :math:`\R^2` (e.g. images), the complete G-steerable kernel spaces for :math:`G \leq \O2`
        is derived in `General E(2)-Equivariant Steerable CNNs <https://arxiv.org/abs/1911.08251>`_.

        During training, in each forward pass the module expands the basis of G-steerable kernels with learned weights
        before calling :func:`torch.nn.functional.conv2d`.
        When :meth:`~torch.nn.Module.eval()` is called, the filter is built with the current trained weights and stored
        for future reuse such that no overhead of expanding the kernel remains.
        
        .. warning ::
            
            When :meth:`~torch.nn.Module.train()` is called, the attributes :attr:`~e2cnn.nn.R2Conv.filter` and
            :attr:`~e2cnn.nn.R2Conv.expanded_bias` are discarded to avoid situations of mismatch with the
            learnable expansion coefficients.
            See also :meth:`e2cnn.nn.R2Conv.train`.
            
            This behaviour can cause problems when storing the :meth:`~torch.nn.Module.state_dict` of a model while in
            a mode and lately loading it in a model with a different mode, as the attributes of the class change.
            To avoid this issue, we recommend converting the model to eval mode before storing or loading the state
            dictionary.
 
 
        The learnable expansion coefficients of the this module can be initialized with the methods in
        :mod:`e2cnn.nn.init`.
        By default, the weights are initialized in the constructors using :func:`~e2cnn.nn.init.generalized_he_init`.
        
        .. warning ::
            
            This initialization procedure can be extremely slow for wide layers.
            In case initializing the model is not required (e.g. before loading the state dict of a pre-trained model)
            or another initialization method is preferred (e.g. :func:`~e2cnn.nn.init.deltaorthonormal_init`), the
            parameter ``initialize`` can be set to ``False`` to avoid unnecessary overhead.
        
        
        The parameters ``basisexpansion``, ``sigma``, ``frequencies_cutoff``, ``rings`` and ``maximum_offset`` are
        optional parameters used to control how the basis for the filters is built, how it is sampled on the filter
        grid and how it is expanded to build the filter. We suggest to keep these default values.
        
        
        Args:
            in_type (FieldType): the type of the input field, specifying its transformation law
            out_type (FieldType): the type of the output field, specifying its transformation law
            kernel_size (int): the size of the (square) filter
            padding (int, optional): implicit zero paddings on both sides of the input. Default: ``0``
            padding_mode(str, optional): ``zeros``, ``reflect``, ``replicate`` or ``circular``. Default: ``zeros``
            stride (int, optional): the stride of the kernel. Default: ``1``
            dilation (int, optional): the spacing between kernel elements. Default: ``1``
            groups (int, optional): number of blocked connections from input channels to output channels.
                                    It allows depthwise convolution. When used, the input and output types need to be
                                    divisible in ``groups`` groups, all equal to each other.
                                    Default: ``1``.
            bias (bool, optional): Whether to add a bias to the output (only to fields which contain a
                    trivial irrep) or not. Default ``True``
            basisexpansion (str, optional): the basis expansion algorithm to use
            sigma (list or float, optional): width of each ring where the bases are sampled. If only one scalar
                    is passed, it is used for all rings.
            frequencies_cutoff (callable or float, optional): function mapping the radii of the basis elements to the
                    maximum frequency accepted. If a float values is passed, the maximum frequency is equal to the
                    radius times this factor. By default (``None``), a more complex policy is used.
            rings (list, optional): radii of the rings where to sample the bases
            maximum_offset (int, optional): number of additional (aliased) frequencies in the intertwiners for finite
                    groups. By default (``None``), all additional frequencies allowed by the frequencies cut-off
                    are used.
            recompute (bool, optional): if ``True``, recomputes a new basis for the equivariant kernels.
                    By Default (``False``), it  caches the basis built or reuse a cached one, if it is found.
            basis_filter (callable, optional): function which takes as input a descriptor of a basis element
                    (as a dictionary) and returns a boolean value: whether to preserve (``True``) or discard (``False``)
                    the basis element. By default (``None``), no filtering is applied.
            init (str, optional): Initialization to use. Can be ``"he"`` for the generalized He init,
                    ``"delta"`` for the delta-orthonormal init, or ``None`` to not initialize at all.
        
        Attributes:
            
            ~.weights (torch.Tensor): the learnable parameters which are used to expand the kernel
            ~.filter (torch.Tensor): the convolutional kernel obtained by expanding the parameters
                                    in :attr:`~e2cnn.nn.R2Conv.weights`
            ~.bias (torch.Tensor): the learnable parameters which are used to expand the bias, if ``bias=True``
            ~.expanded_bias (torch.Tensor): the equivariant bias which is summed to the output, obtained by expanding
                                    the parameters in :attr:`~e2cnn.nn.R2Conv.bias`
        
        """
        super().__init__(in_type, out_type,
                         kernel_size,
                         padding=padding,
                         padding_mode=padding_mode,
                         stride=stride,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         basisexpansion=basisexpansion,
                         sigma=sigma,
                         frequencies_cutoff=frequencies_cutoff,
                         rings=rings,
                         maximum_offset=maximum_offset,
                         recompute=recompute,
                         normalize_basis=normalize_basis,
                         basis_filter=basis_filter,
                         init=init
                         )

    def _compute_basis_params(self,
                              frequencies_cutoff: Union[float, Callable[[float], float]] = None,
                              rings: List[float] = None,
                              sigma: List[float] = None,
                              custom_basis_filter: Callable[[dict], bool] = None,
                              normalize_basis: bool = True,
                              **kwargs
                              ):

        grid = get_grid_coords(self.kernel_size, self.dilation)
        # compute the coordinates of the centers of the cells in the grid where the filter is sampled
        max_radius = np.sqrt((grid **2).sum(0)).max()
        # max_radius = kernel_size // 2

        # by default, the number of rings equals half of the filter size
        if rings is None:
            n_rings = math.ceil(self.kernel_size / 2)
            # if self.group.order() > 0:
            #     # compute the number of edges of the polygon inscribed in the filter (which is a square)
            #     # whose points stay inside the filter under the action of the group
            #     # the number of edges is lcm(group's order, 4)
            #     n_edges = self.group.order()
            #     while n_edges % 4 > 0:
            #         n_edges *= 2
            #     # the largest ring we can sample has radius equal to the circumradius of the polygon described above
            #     n_rings /= math.cos(math.pi/n_edges)

            # n_rings = s // 2 + 1

            # rings = torch.linspace(1 - s % 2, s // 2, n_rings)
            rings = torch.linspace(0, (self.kernel_size - 1) // 2, n_rings) * self.dilation
            rings = rings.tolist()

        assert all([max_radius >= r >= 0 for r in rings])

        if sigma is None:
            sigma = [0.6] * (len(rings) - 1) + [0.4]
            for i, r in enumerate(rings):
                if r == 0.:
                    sigma[i] = 0.005

        elif isinstance(sigma, float):
            sigma = [sigma] * len(rings)

        # TODO - use a string name for this setting
        if frequencies_cutoff is None:
            frequencies_cutoff = -1.

        if isinstance(frequencies_cutoff, float):
            if frequencies_cutoff == -3:
                frequencies_cutoff = _manual_fco3(self.kernel_size // 2)
            elif frequencies_cutoff == -2:
                frequencies_cutoff = _manual_fco2(self.kernel_size // 2)
            elif frequencies_cutoff == -1:
                frequencies_cutoff = _manual_fco1(self.kernel_size // 2)
            else:
                frequencies_cutoff = lambda r, fco=frequencies_cutoff: fco * r

        # check if the object is a callable function
        assert callable(frequencies_cutoff)

        maximum_frequency = int(max(frequencies_cutoff(r) for r in rings))

        fco_filter = bandlimiting_filter(frequencies_cutoff)

        if custom_basis_filter is not None:
            basis_filter = lambda d, custom_basis_filter=custom_basis_filter, fco_filter=fco_filter: (custom_basis_filter(d) and fco_filter(d))
        else:
            basis_filter = fco_filter

        return grid, basis_filter, {
            "rings": rings,
            "sigma": sigma,
            "maximum_frequency": maximum_frequency,
            "method": "kernel",
            "normalize": normalize_basis,
        }


class R2GeneralConv(R2General):
    
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 num_neighbors: int,
                 in_grid: Grid,
                 out_grid: Grid = None,
                 groups: int = 1,
                 bias: bool = True,
                 sigma: Union[List[float], float] = None,
                 frequencies_cutoff: Union[float, Callable[[float], int]] = None,
                 rings: List[float] = None,
                 maximum_offset: int = None,
                 recompute: bool = False,
                 normalize_basis: bool = True,
                 basis_filter: Callable[[dict], bool] = None,
                 init: str = "he"
                 ):
        super().__init__(in_type, out_type,
                         num_neighbors,
                         in_grid, out_grid,
                         groups=groups,
                         bias=bias,
                         sigma=sigma,
                         frequencies_cutoff=frequencies_cutoff,
                         rings=rings,
                         maximum_offset=maximum_offset,
                         recompute=recompute,
                         normalize_basis=normalize_basis,
                         basis_filter=basis_filter,
                         init=init
                         )

    def _compute_basis_params(self,
                              frequencies_cutoff: Union[float, Callable[[float], float]] = None,
                              rings: List[float] = None,
                              sigma: List[float] = None,
                              custom_basis_filter: Callable[[dict], bool] = None,
                              normalize_basis: bool = True,
                              **kwargs
                              ):
        # by default, the number of rings equals half of the filter size
        if rings is None:
            # this is of course only an approximation and how well it works
            # depends on the grid. If the grid is very irregular, you should
            # probably set rings by hand
            effective_kernel_size = math.sqrt(self.num_neighbors)
            n_rings = math.ceil(effective_kernel_size / 2)
            rings = torch.linspace(0, (effective_kernel_size - 1) // 2, n_rings)
            rings = rings.tolist()

        if sigma is None:
            sigma = [0.6] * (len(rings) - 1) + [0.4]
            for i, r in enumerate(rings):
                if r == 0.:
                    sigma[i] = 0.005

        elif isinstance(sigma, float):
            sigma = [sigma] * len(rings)

        # TODO - use a string name for this setting
        if frequencies_cutoff is None:
            frequencies_cutoff = -1.

        if isinstance(frequencies_cutoff, float):
            if frequencies_cutoff == -3:
                frequencies_cutoff = _manual_fco3(max(rings))
            elif frequencies_cutoff == -2:
                frequencies_cutoff = _manual_fco2(max(rings))
            elif frequencies_cutoff == -1:
                frequencies_cutoff = _manual_fco1(max(rings))
            else:
                frequencies_cutoff = lambda r, fco=frequencies_cutoff: fco * r

        # check if the object is a callable function
        assert callable(frequencies_cutoff)

        maximum_frequency = int(max(frequencies_cutoff(r) for r in rings))

        fco_filter = bandlimiting_filter(frequencies_cutoff)

        if custom_basis_filter is not None:
            basis_filter = lambda d, custom_basis_filter=custom_basis_filter, fco_filter=fco_filter: (custom_basis_filter(d) and fco_filter(d))
        else:
            basis_filter = fco_filter

        return basis_filter, {
            "rings": rings,
            "sigma": sigma,
            "maximum_frequency": maximum_frequency,
            "method": "kernel",
            "normalize": normalize_basis,
        }


def bandlimiting_filter(frequency_cutoff: Union[float, Callable[[float], float]]) -> Callable[[dict], bool]:
    r"""

    Returns a method which takes as input the attributes (as a dictionary) of a basis element and returns a boolean
    value: whether to preserve that element (True) or not (False)
    
    If the parameter ``frequency_cutoff`` is a scalar value, the maximum frequency allowed at a certain radius is
    proportional to the radius itself. In thi case, the parameter ``frequency_cutoff`` is the factor controlling this
    proportionality relation.
    
    If the parameter ``frequency_cutoff`` is a callable, it needs to take as input a radius (a scalar value) and return
    the maximum frequency which can be sampled at that radius.

    Args:
        frequency_cutoff (float): factor controlling the bandlimiting

    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """
    
    if isinstance(frequency_cutoff, float):
        frequency_cutoff = lambda r, fco=frequency_cutoff: r * frequency_cutoff
    
    def bl_filter(attributes: dict) -> bool:
        return math.fabs(attributes["frequency"]) <= frequency_cutoff(attributes["radius"])
    
    return bl_filter


def _manual_fco3(max_radius: float) -> Callable[[float], float]:
    r"""

    Returns a method which takes as input the radius of a ring and returns the maximum frequency which can be sampled
    on that ring.

    Args:
        max_radius (float): radius of the last ring touching the border of the grid

    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """
    
    def bl_filter(r: float) -> float:
        max_freq = 0 if r == 0. else 1 if r == max_radius else 2
        return max_freq
    
    return bl_filter


def _manual_fco2(max_radius: float) -> Callable[[float], float]:
    r"""

    Returns a method which takes as input the radius of a ring and returns the maximum frequency which can be sampled
    on that ring.

    Args:
        max_radius (float): radius of the last ring touching the border of the grid

    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """
    
    def bl_filter(r: float) -> float:
        max_freq = 0 if r == 0. else min(2 * r, 1 if r == max_radius else 2 * r - (r + 1) % 2)
        return max_freq
    
    return bl_filter


def _manual_fco1(max_radius: float) -> Callable[[float], float]:
    r"""

    Returns a method which takes as input the radius of a ring and returns the maximum frequency which can be sampled
    on that ring.

    Args:
        max_radius (float): radius of the last ring touching the border of the grid

    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """
    
    def bl_filter(r: float) -> float:
        max_freq = 0 if r == 0. else min(2 * r, 2 if r == max_radius else 2 * r - (r + 1) % 2)
        return max_freq
    
    return bl_filter
