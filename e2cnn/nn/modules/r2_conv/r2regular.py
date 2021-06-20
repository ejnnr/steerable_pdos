from abc import abstractmethod
from typing import Tuple, Dict, Callable, Union
import math
import torch
import numpy as np
from torch.nn.functional import conv2d, pad
from .basisexpansion_blocks import BlocksBasisExpansion
from e2cnn.nn import GeometricTensor, FieldType
from .r2layer import R2Layer

class R2Regular(R2Layer):
    def __init__(self,
               in_type: FieldType,
               out_type: FieldType,
               kernel_size: int,
               padding: int = 0,
               stride: int = 1,
               dilation: int = 1,
               padding_mode: str = 'zeros',
               **kwargs
               ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode

        if isinstance(padding, tuple) and len(padding) == 2:
            _padding = padding
        elif isinstance(padding, int):
            _padding = (padding, padding)
        else:
            raise ValueError('padding needs to be either an integer or a tuple containing two integers but {} found'.format(padding))

        padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in padding_modes:
            raise ValueError("padding_mode must be one of [{}], but got padding_mode='{}'".format(padding_modes, padding_mode))
        self._reversed_padding_repeated_twice = tuple(x for x in reversed(_padding) for _ in range(2))

        super().__init__(in_type, out_type, **kwargs)

    def _forward(self, input, filter, bias):
        if self.padding_mode == 'zeros':
            output = conv2d(input.tensor, filter,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            groups=self.groups,
                            bias=bias)
        else:
            output = conv2d(pad(input.tensor, self._reversed_padding_repeated_twice, self.padding_mode),
                            filter,
                            stride=self.stride,
                            dilation=self.dilation,
                            padding=(0,0),
                            groups=self.groups,
                            bias=bias)

        return GeometricTensor(output, self.out_type)

    @abstractmethod
    def _compute_basis_params(self, custom_basis_filter, **kwargs) -> Tuple[Union[np.ndarray, int], Callable[[Dict], bool], Dict]:
        """This is the function that needs to be overwritten for a fully specified layer.
        It should return the grid, the basis_filter and a dict of params to pass to the
        basisexpansion module."""
        pass

    def _init_basisexpansion(self, in_type, out_type, basisexpansion, basis_filter, recompute, maximum_offset, **kwargs):
        grid, basis_filter, params = self._compute_basis_params(custom_basis_filter=basis_filter, **kwargs)

        # BasisExpansion: submodule which takes care of building the filter
        self._basisexpansion = None

        # notice that `in_type` is used instead of `self.in_type` such that it works also when `groups > 1`
        if basisexpansion == 'blocks':
            self._basisexpansion = BlocksBasisExpansion(in_type, out_type,
                                                        grid,
                                                        recompute=recompute,
                                                        maximum_offset=maximum_offset,
                                                        basis_filter=basis_filter,
                                                        **params)

        else:
            raise ValueError('Basis Expansion algorithm "%s" not recognized' % basisexpansion)

    def extra_repr(self):
        s = ('{in_type}, {out_type}, kernel_size={kernel_size}, stride={stride}')
        if self.padding != 0 and self.padding != (0, 0):
            s += ', padding={padding}'
        if self.dilation != 1 and self.dilation != (1, 1):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def expand_parameters(self):
        _filter = self.basisexpansion(self.weights)
        _filter = _filter.reshape(_filter.shape[0], _filter.shape[1], self.kernel_size, self.kernel_size)
        _bias = self._expand_bias()

        return _filter, _bias

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size

        b, c, hi, wi = input_shape

        ho = math.floor((hi + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
        wo = math.floor((wi + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)

        return b, self.out_type.size, ho, wo

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.Conv2d` module and set to "eval" mode.

        """

        # set to eval mode so the filter and the bias are updated with the current
        # values of the weights
        self.eval()
        _filter = self.filter
        _bias = self.expanded_bias

        if self.padding_mode not in ['zeros']:
            x, y = torch.__version__.split('.')[:2]
            if int(x) < 1 or int(y) < 5:
                if self.padding_mode == 'circular':
                    raise ImportError(
                        "'{}' padding mode had some issues in old `torch` versions. Therefore, we only support conversion from version 1.5 but only version {} is installed.".format(
                            self.padding_mode, torch.__version__
                        )
                    )

                else:
                    raise ImportError(
                        "`torch` supports '{}' padding mode only from version 1.5 but only version {} is installed.".format(
                            self.padding_mode, torch.__version__
                        )
                    )

        # build the PyTorch Conv2d module
        has_bias = self.bias is not None
        conv = torch.nn.Conv2d(self.in_type.size,
                               self.out_type.size,
                               self.kernel_size,
                               padding=self.padding,
                               padding_mode=self.padding_mode,
                               stride=self.stride,
                               dilation=self.dilation,
                               groups=self.groups,
                               bias=has_bias)

        # set the filter and the bias
        conv.weight.data = _filter.data
        if has_bias:
            conv.bias.data = _bias.data

        return conv

    def check_equivariance(self, atol: float = 0.1, rtol: float = 0.1, assertion: bool = True, verbose: bool = True):

        # np.set_printoptions(precision=5, threshold=30 *self.in_type.size**2, suppress=False, linewidth=30 *self.in_type.size**2)

        feature_map_size = 33
        last_downsampling = 5
        first_downsampling = 5

        initial_size = (feature_map_size * last_downsampling - 1 + self.kernel_size) * first_downsampling

        c = self.in_type.size

        import matplotlib.image as mpimg
        from skimage.measure import block_reduce
        from skimage.transform import resize

        x = mpimg.imread('../group/testimage.jpeg').transpose((2, 0, 1))[np.newaxis, 0:c, :, :]

        x = resize(
            x,
            (x.shape[0], x.shape[1], initial_size, initial_size),
            anti_aliasing=True
        )

        x = x / 255.0 - 0.5

        if x.shape[1] < c:
            to_stack = [x for i in range(c // x.shape[1])]
            if c % x.shape[1] > 0:
                to_stack += [x[:, :(c % x.shape[1]), ...]]

            x = np.concatenate(to_stack, axis=1)

        x = GeometricTensor(torch.FloatTensor(x), self.in_type)

        def shrink(t: GeometricTensor, s) -> GeometricTensor:
            return GeometricTensor(torch.FloatTensor(block_reduce(t.tensor.detach().numpy(), s, func=np.mean)), t.type)

        errors = []

        for el in self.space.testing_elements:

            out1 = self(shrink(x, (1, 1, 5, 5))).transform(el).tensor.detach().numpy()
            out2 = self(shrink(x.transform(el), (1, 1, 5, 5))).tensor.detach().numpy()

            out1 = block_reduce(out1, (1, 1, 5, 5), func=np.mean)
            out2 = block_reduce(out2, (1, 1, 5, 5), func=np.mean)

            b, c, h, w = out2.shape

            center_mask = np.zeros((2, h, w))
            center_mask[1, :, :] = np.arange(0, w) - w / 2
            center_mask[0, :, :] = np.arange(0, h) - h / 2
            center_mask[0, :, :] = center_mask[0, :, :].T
            center_mask = center_mask[0, :, :] ** 2 + center_mask[1, :, :] ** 2 < (h / 4) ** 2

            out1 = out1[..., center_mask]
            out2 = out2[..., center_mask]

            out1 = out1.reshape(-1)
            out2 = out2.reshape(-1)

            errs = np.abs(out1 - out2)

            esum = np.maximum(np.abs(out1), np.abs(out2))
            esum[esum == 0.0] = 1

            relerr = errs / esum

            if verbose:
                print(el, relerr.max(), relerr.mean(), relerr.var(), errs.max(), errs.mean(), errs.var())

            tol = rtol * esum + atol

            if np.any(errs > tol) and verbose:
                print(out1[errs > tol])
                print(out2[errs > tol])
                print(tol[errs > tol])

            if assertion:
                assert np.all(errs < tol), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(el, errs.max(), errs.mean(), errs.var())

            errors.append((el, errs.mean()))

        return errors

        # init.deltaorthonormal_init(self.weights.data, self.basisexpansion)
        # filter = self.basisexpansion()
        # center = self.s // 2
        # filter = filter[..., center, center]
        # assert torch.allclose(torch.eye(filter.shape[1]), filter.t() @ filter, atol=3e-7)

    @property
    def is_irregular(self):
        return False