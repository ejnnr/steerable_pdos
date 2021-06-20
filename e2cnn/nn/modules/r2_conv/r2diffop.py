
import warnings

from torch.functional import norm

from e2cnn.diffops.utils import (
    load_cache,
    store_cache,
    symmetric_points,
    required_points,
    largest_possible_order,
    guaranteed_accuracy
)

from e2cnn.nn import FieldType, GeometricTensor, Grid

from .r2regular import R2Regular
from .r2general import R2General
from .utils import get_grid_coords

from typing import Callable, Union, Tuple, List, Dict


__all__ = ["R2Diffop"]


class R2Diffop(R2Regular):

    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 kernel_size: int = None,
                 accuracy: int = None,
                 padding: int = 0,
                 stride: int = 1,
                 dilation: int = 1,
                 padding_mode: str = 'zeros',
                 groups: int = 1,
                 bias: bool = True,
                 basisexpansion: str = 'blocks',
                 maximum_order: int = None,
                 maximum_partial_order: int = None,
                 maximum_offset: int = None,
                 maximum_power: int = None,
                 special_regular_basis: bool = False,
                 rotate90: bool = False,
                 recompute: bool = False,
                 normalize_basis: bool = True,
                 angle_offset: float = None,
                 basis_filter: Callable[[dict], bool] = None,
                 init: str = "he",
                 cache: Union[bool, str] = True,
                 rbffd: bool = False,
                 smoothing: float = None,
                 restrict_kernel_size: bool = False,
                 ):
        if cache and cache != "store":
            # Load the cached lambdas for RBFs if they exist
            load_cache()

        if special_regular_basis and special_basis_applicable(in_type, out_type) and init is not None:
            # if we don't have regular -> regular blocks,
            # no need for this init, since we won't use the special regular basis
            # anyway
            init = "regular"

        if maximum_partial_order is not None:
            if maximum_order is None:
                maximum_order = 2 * maximum_partial_order
            if not special_regular_basis:
                warnings.warn("Using a partial order filter without the special regular basis."
                              "This will probably lead to a smaller basis than desired.")

        # out of kernel_size, accuracy and maximum_order, exactly two must be known,
        # the third one can then be determined automatically.
        # To provide sane defaults, we will also allow only kernel_size or maximum_order
        # to be set, in that case accuracy will become 2.
        if kernel_size is None:
            assert maximum_order is not None
            if accuracy is None:
                accuracy = 2
            # TODO: Ideally, we should look at the basis, maybe the maximum_order isn't
            # reached (e.g. if it is odd but all basis diffops are even). In that case,
            # we could perhaps get away with a smaller kernel
            kernel_size = required_points(maximum_order, accuracy)
        elif maximum_order is None:
            assert kernel_size is not None
            if accuracy is None:
                accuracy = 2 if (kernel_size > 1) else 1
            maximum_order = largest_possible_order(kernel_size, accuracy)
            if maximum_order < 2:
                warnings.warn(f"Maximum order is only {maximum_order} for kernel size "
                              f"{kernel_size} and desired accuracy {accuracy}. This may "
                              "lead to a small basis. If this is unintentional, consider "
                              "increasing the kernel size.")
        elif accuracy is None:
            if kernel_size < required_points(maximum_order, 2):
                warnings.warn(f"Small kernel size: {kernel_size} x {kernel_size} kernel "
                              f"is used for differential operators of order up to {maximum_order}. "
                              "This may lead to bad approximations, consider using a larger kernel "
                              "or setting the desired accuracy instead of the kernel size.")
        else:
            # all three are set
            raise ValueError("At most two of kernel size, maximum order and accuracy can bet set, "
                             "see documentation for details.")

        super().__init__(in_type, out_type,
                         kernel_size,
                         padding=padding,
                         padding_mode=padding_mode,
                         stride=stride,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         basisexpansion=basisexpansion,
                         maximum_offset=maximum_offset,
                         maximum_order=maximum_order,
                         maximum_partial_order=maximum_partial_order,
                         maximum_power=maximum_power,
                         special_regular_basis=special_regular_basis,
                         rotate90=rotate90,
                         recompute=recompute,
                         normalize_basis=normalize_basis,
                         angle_offset=angle_offset,
                         basis_filter=basis_filter,
                         init=init,
                         rbffd=rbffd,
                         smoothing=smoothing,
                         restrict_kernel_size=restrict_kernel_size,
                         )

        if cache and cache != "load":
            store_cache()

    def _compute_basis_params(self,
                            custom_basis_filter: Callable[[dict], bool],
                            maximum_order: int,
                            maximum_partial_order: int,
                            maximum_power: int,
                            special_regular_basis: bool,
                            rotate90: bool,
                            rbffd: bool,
                            smoothing: float,
                            restrict_kernel_size: bool,
                            normalize_basis: bool,
                            angle_offset: float,
                            **kwargs
                            ):

        if rbffd:
            if smoothing is not None:
                warnings.warn("RBF-FD flag has no effect when smoothing is used")
            # compute the coordinates of the centers of the cells in the grid where the filter is sampled
            grid = get_grid_coords(self.kernel_size, self.dilation)
        else:
            # For FD, we don't pass on points but instead just a 1D list of coordinates.
            # These should be arranged symmetrically around 0, which is what the following
            # helper function does (essentially just a 1D analogon of get_grid_coords)
            grid = symmetric_points(self.kernel_size, self.dilation)

        basis_filter = compute_basis_filter(maximum_order, custom_basis_filter)
        params = param_dict(maximum_order, maximum_power, maximum_partial_order, special_regular_basis)
        params["smoothing"] = smoothing
        params["rotate90"] = rotate90
        params["normalize"] = normalize_basis
        params["angle_offset"] = angle_offset
        if restrict_kernel_size:
            # HACK: This works for important cases, but it's not really what we
            # want in general
            params["accuracy"] = 2

        return grid, basis_filter, params


class R2GeneralDiffop(R2General):

    def __init__(self,
               in_type: FieldType,
               out_type: FieldType,
               num_neighbors: int,
               in_grid: Grid,
               out_grid: Grid = None,
               groups: int = 1,
               bias: bool = True,
               maximum_order: int = 2,
               maximum_partial_order: int = None,
               maximum_offset: int = None,
               maximum_power: int = None,
               recompute: bool = False,
               normalize_basis: bool = True,
               basis_filter: Callable[[dict], bool] = None,
               init: str = "he",
               cache: Union[bool, str] = True,
               ):

        if cache and cache != "store":
            # Load the cached lambdas for RBFs if they exist
            load_cache()

        super().__init__(in_type, out_type,
                         num_neighbors,
                         in_grid, out_grid,
                         groups=groups,
                         bias=bias,
                         maximum_offset=maximum_offset,
                         maximum_order=maximum_order,
                         maximum_partial_order=maximum_partial_order,
                         maximum_power=maximum_power,
                         recompute=recompute,
                         normalize_basis=normalize_basis,
                         basis_filter=basis_filter,
                         init=init
                         )

        if cache and cache != "load":
            store_cache()

    def _compute_basis_params(self,
                            custom_basis_filter: Callable[[Dict], bool],
                            maximum_order: int,
                            maximum_partial_order: int,
                            maximum_power: int,
                            normalize_basis: bool,
                            **kwargs
                            ):

        params = param_dict(maximum_order, maximum_power, maximum_partial_order)
        params["normalize"] = normalize_basis
        basis_filter = compute_basis_filter(maximum_order, custom_basis_filter)
        return basis_filter, params

def compute_basis_filter(maximum_order: int, custom_basis_filter: Callable[[Dict], bool]):
    filters = [order_filter(maximum_order)]
    if custom_basis_filter is not None:
        filters.append(custom_basis_filter)

    return lambda d: all(f(d) for f in filters)

def param_dict(maximum_order: int, maximum_power: int, maximum_partial_order: int, special_regular_basis: bool = False):
    # special_regular_basis has a default value because it won't be passed
    # by R2GeneralDiffop for now
    if maximum_power is not None:
        maximum_power = min(maximum_power, maximum_order // 2)
    else:
        maximum_power = maximum_order // 2
    return {
        # to guarantee that all relevant tensor products
        # are generated, we need Laplacian powers up to
        # half the maximum order. Anything higher would be
        # discarded anyways by the basis_filter
        "max_power": maximum_power,
        # frequencies higher than than the maximum order will be discarded anyway
        "maximum_frequency": maximum_order,
        "method": "diffop",
        "special_regular_basis": special_regular_basis,
        "maximum_partial_order": maximum_partial_order,
    }

def order_filter(maximum_order: int) -> Callable[[dict], bool]:
    return lambda attr: attr["order"] <= maximum_order

def special_basis_applicable(type1: FieldType, type2: FieldType):
    if contains_regular(type1) and (contains_regular(type2) or contains_trivial(type2)):
        return True
    if contains_regular(type2) and (contains_regular(type1) or contains_trivial(type1)):
        return True
    return False

def contains_regular(type: FieldType):
    return any(r.name == "regular" for r in type.representations)

def contains_trivial(type: FieldType):
    return any(r.name == "irrep_0" for r in type.representations)