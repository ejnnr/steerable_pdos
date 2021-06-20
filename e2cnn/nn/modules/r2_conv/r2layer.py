from abc import ABC, abstractmethod
from typing import Callable, Union, Tuple, List, Dict, Optional

from e2cnn.nn.init import deltaorthonormal_init, generalized_he_init, regular_init
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor
from e2cnn.gspaces import *

from ..equivariant_module import EquivariantModule

from .basisexpansion import BasisExpansion

import torch
from torch.nn import Parameter

class R2Layer(EquivariantModule):

    def __init__(self,
               in_type: FieldType,
               out_type: FieldType,
               groups: int = 1,
               bias: bool = True,
               basisexpansion: str = 'blocks',
               recompute: bool = False,
               maximum_offset: int = None,
               basis_filter: Callable[[dict], bool] = None,
               init: Optional[str] = "he",
               **kwargs
               ):
        """Abstract base class for convolutional and PDO layers on R2."""
        assert in_type.gspace == out_type.gspace
        assert isinstance(in_type.gspace, GeneralOnR2)

        super().__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = out_type

        self.groups = groups

        if groups > 1:
            # Check the input and output classes can be split in `groups` groups, all equal to each other
            # first, check that the number of fields is divisible by `groups`
            assert len(in_type) % groups == 0
            assert len(out_type) % groups == 0
            in_size = len(in_type) // groups
            out_size = len(out_type) // groups

            # then, check that all groups are equal to each other, i.e. have the same types in the same order
            assert all(in_type.representations[i] == in_type.representations[i % in_size] for i in range(len(in_type)))
            assert all(out_type.representations[i] == out_type.representations[i % out_size] for i in range(len(out_type)))

            # finally, retrieve the type associated to a single group in input.
            # this type will be used to build a smaller kernel basis and a smaller filter
            # as in PyTorch, to build a filter for grouped convolution, we build a filter which maps from one input
            # group to all output groups. Then, PyTorch's standard convolution routine interpret this filter as `groups`
            # different filters, each mapping an input group to an output group.
            in_type = in_type.index_select(list(range(in_size)))

        if bias:
            # bias can be applied only to trivial irreps inside the representation
            # to apply bias to a field we learn a bias for each trivial irreps it contains
            # and, then, we transform it with the change of basis matrix to be able to apply it to the whole field
            # this is equivalent to transform the field to its irreps through the inverse change of basis,
            # sum the bias only to the trivial irrep and then map it back with the change of basis

            # count the number of trivial irreps
            trivials = 0
            for r in self.out_type:
                for irr in r.irreps:
                    if self.out_type.fibergroup.irreps[irr].is_trivial():
                        trivials += 1

            # if there is at least 1 trivial irrep
            if trivials > 0:

                # matrix containing the columns of the change of basis which map from the trivial irreps to the
                # field representations. This matrix allows us to map the bias defined only over the trivial irreps
                # to a bias for the whole field more efficiently
                bias_expansion = torch.zeros(self.out_type.size, trivials)

                p, c = 0, 0
                for r in self.out_type:
                    pi = 0
                    for irr in r.irreps:
                        irr = self.out_type.fibergroup.irreps[irr]
                        if irr.is_trivial():
                            bias_expansion[p:p+r.size, c] = torch.tensor(r.change_of_basis[:, pi])
                            c += 1
                        pi += irr.size
                    p += r.size

                self.register_buffer("bias_expansion", bias_expansion)
                self.bias = Parameter(torch.zeros(trivials), requires_grad=True)
                self.register_buffer("expanded_bias", torch.zeros(out_type.size), persistent=False)
            else:
                self.bias = None
                self.expanded_bias = None
        else:
            self.bias = None
            self.expanded_bias = None

        # notice that `in_type` is used instead of `self.in_type` such that it works also when `groups > 1`
        self._init_basisexpansion(in_type, out_type,
                                  basis_filter=basis_filter,
                                  recompute=recompute,
                                  basisexpansion=basisexpansion,
                                  maximum_offset=maximum_offset,
                                  **kwargs)

        if self.basisexpansion.dimension() == 0:
            raise ValueError('''
                The basis for the steerable filter is empty!
                Tune the `frequencies_cutoff`, `kernel_size`, `rings`, `sigma` or `basis_filter` parameters to allow
                for a larger basis.
            ''')

        self.weights = Parameter(torch.zeros(self.basisexpansion.dimension()), requires_grad=True)

        if init == "he":
            # by default, the weights are initialized with a generalized form of He's weight initialization
            generalized_he_init(self.weights.data, self.basisexpansion)
        elif init == "delta":
            deltaorthonormal_init(self.weights.data, self.basisexpansion)
        elif init == "regular":
            regular_init(self.weights.data, self.basisexpansion)
        elif init is not None:
            raise ValueError("Init must be 'he', 'delta', 'regular' or None.")

        # HACK: when the weights are loaded from a checkpoint, we need to remove the
        # filter and bias expansion, because they probably don't match the loaded weights.
        # This wouldn't be an issue in training mode, but if the model is in eval when
        # the state dict is loaded, this is necessary.
        # Unfortunately, _register_state_dict_hook is an internal function,
        # so this may not work in future versions of Pytorch
        self._register_state_dict_hook(lambda obj, *args: obj._remove_precomputed_filter())

    def _remove_precomputed_filter(self):
        if hasattr(self, "filter"):
            del self.filter
        if hasattr(self, "expanded_bias"):
            del self.expanded_bias

    @abstractmethod
    def _init_basisexpansion(self,
                           in_type: FieldType,
                           out_type: FieldType,
                           recompute: bool,
                           basis_filter: Callable[[Dict], bool],
                           basisexpansion: str,
                           **kwargs):
        """Initialize the basis expansion module and the filter buffer.

        Should create a self._basisexpansion attribute and a self.filter buffer."""
        pass

    @property
    def basisexpansion(self) -> BasisExpansion:
        r"""
        Submodule which takes care of building the filter.

        It uses the learnt ``weights`` to expand a basis and returns a filter in the usual form used by conventional
        convolutional modules.
        It uses the learned ``weights`` to expand the kernel in the G-steerable basis and returns it in the shape
        :math:`(c_\text{out}, c_\text{in}, s^2)`, where :math:`s` is the ``kernel_size``.

        """
        return self._basisexpansion

    def _expand_bias(self) -> Optional[torch.Tensor]:
        """Helper function which expands the bias, to be used in ``expand_parameters`` implementation."""
        if self.bias is None:
            return None
        else:
            return self.bias_expansion @ self.bias

    @abstractmethod
    def expand_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Expand the filter in terms of the :attr:`e2cnn.nn.R2Conv.weights` and the
        expanded bias in terms of :class:`e2cnn.nn.R2Conv.bias`.

        Returns:
            the expanded filter and bias

        """
        pass

    @abstractmethod
    def _forward(self, input: GeometricTensor, filter, bias):
        """Internal method which should implement the forward pass for a given filter and bias.

        This method is called inside forward(). When subclassing this class, you can usually just
        override this method."""
        pass

    def forward(self, input: GeometricTensor):
        r"""
        Convolve the input with the expanded filter and bias.

        Args:
            input (GeometricTensor): input feature field transforming according to ``in_type``

        Returns:
            output feature field transforming according to ``out_type``

        """

        assert input.type == self.in_type

        if self.training or self.is_irregular:
            # During training, we need to compute these dynamically for autograd.
            # And on irregular grids, memory is usually the bigger issue, so we also
            # don't store filters
            _filter, _bias = self.expand_parameters()
        elif hasattr(self, "filter") and hasattr(self, "expanded_bias"):
            # otherwise, we can use the existing filter
            _filter = self.filter
            _bias = self.expanded_bias
        else:
            # this happens if a model is put into eval mode and then the state dict is loaded
            _filter, _bias = self.expand_parameters()
            self.filter = _filter
            self.expanded_bias = _bias

        return self._forward(input, _filter, _bias)

    def train(self, mode=True):
        r"""

        If ``mode=True``, the method sets the module in training mode and discards the :attr:`~e2cnn.nn.R2Conv.filter`
        and :attr:`~e2cnn.nn.R2Conv.expanded_bias` attributes.

        If ``mode=False``, it sets the module in evaluation mode. Moreover, the method builds the filter and the bias using
        the current values of the trainable parameters and store them in :attr:`~e2cnn.nn.R2Conv.filter` and
        :attr:`~e2cnn.nn.R2Conv.expanded_bias` such that they are not recomputed at each forward pass.

        .. warning ::

            This behaviour can cause problems when storing the :meth:`~torch.nn.Module.state_dict` of a model while in
            a mode and lately loading it in a model with a different mode, as the attributes of this class change.
            To avoid this issue, we recommend converting the model to eval mode before storing or loading the state
            dictionary.

        Args:
            mode (bool, optional): whether to set training mode (``True``) or evaluation mode (``False``).
                                   Default: ``True``.

        """

        if mode:
            # TODO thoroughly check this is not causing problems
            self._remove_precomputed_filter()
        elif self.training and not self.is_irregular:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            # Also don't store filters for irregular grids because they take up lots of memory

            _filter, _bias = self.expand_parameters()

            if isinstance(_filter, torch.Tensor):
                self.register_buffer("filter", _filter, persistent=False)
            else:
                self.filter = _filter
            if _bias is not None:
                self.register_buffer("expanded_bias", _bias, persistent=False)
            else:
                self.expanded_bias = None

        return super().train(mode)

    def extra_repr(self):
        return []

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')

        main_str = self._get_name() + '('
        if len(extra_lines) == 1:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(extra_lines) + '\n'

        main_str += ')'
        return main_str
    
    @property
    @abstractmethod
    def is_irregular(self):
        pass
