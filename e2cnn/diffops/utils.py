
from typing import List, Union, Iterable, Tuple
import os
import warnings
import pickle

import numpy as np
import sparse
import scipy.special  # type: ignore
from sympy.calculus.finite_diff import finite_diff_weights
from rbf.pde.fd import weight_matrix, weights # type: ignore
from rbf.basis import set_symbolic_to_numeric_method, get_rbf
from rbf.utils import KDTree

set_symbolic_to_numeric_method('lambdify')
phi = get_rbf("phs3")
_gaussian = get_rbf("ga")

_DIFFOP_CACHE = {}
_1D_KERNEL_CACHE = {}

def load_cache():
    if os.path.exists("./.e2cnn_cache/diffops.pickle"):
        print("Loading cached Diffops")
        with open("./.e2cnn_cache/diffops.pickle", "rb") as f:
            _DIFFOP_CACHE = pickle.load(f)
    else:
        print("Diffop cache not found, skipping")

def store_cache():
    os.makedirs("./.e2cnn_cache", exist_ok=True)
    with open("./.e2cnn_cache/diffops.pickle", "w+b") as f:
        pickle.dump(_DIFFOP_CACHE, f)

def rbffd_discretization(
    coefficients: np.ndarray,
    out_coords: np.ndarray,
    stencils: np.ndarray
) -> np.ndarray:
    """Compute the RBF-FD weights for a given Diffop and grids.

    Args:
        coefficients (ndarray): array with the coefficients of x^n, x^{n - 1}y, ..., y^n
          (in that order)
        out_coords (ndarray): 2 x M array with output grid (i.e. points at which the
          derivative is to be found), where M is the number of output points
        stencils (ndarray): 2 x M x num_neighbors array with stencils for each output
          point
    
    Returns:
        (M, num_neighbors) ndarray with the RBF-FD weights for each stencil point
    """
    if isinstance(coefficients, (list, tuple)):
        coefficients = np.array(coefficients)
    assert len(out_coords.shape) == 2
    assert out_coords.shape[0] == 2
    assert len(stencils.shape) == 3
    assert stencils.shape[0] == 2
    assert stencils.shape[1] == out_coords.shape[1]

    M, num_neighbors = stencils.shape[1:]

    key = (coefficients.tobytes(), out_coords.tobytes(), stencils.tobytes())

    if key in _RBFFD_CACHE:
        return _RBFFD_CACHE[key]

    n = len(coefficients) - 1

    nonzero = (coefficients != 0)
    nonzero_indices = [k for k in range(n + 1) if nonzero[k]]

    diffs = [[n - k, k] for k in nonzero_indices]
    if not diffs:
        # we have the zero operator
        return np.zeros((M, num_neighbors))

    # the RBF library expects the spatial dimension at the end
    stencils = stencils.transpose(1, 2, 0)
    out_coords = out_coords.T

    out = weights(out_coords, stencils, diffs, coefficients[nonzero], phi=phi)

    if np.any(np.isnan(out)):
        raise Exception(f"NaNs encountered while discretizing diffop {display_diffop(coefficients)} on {num_neighbors} points.\n"
                        f"Diffop passed to RBF: {diffs} with coefficients {coefficients[nonzero]}")
    if np.all(out == 0):
        warnings.warn(f"Zero filter encountered while discretizing diffop {display_diffop(coefficients)} on {num_neighbors} points. "
                    "This might indicate that the kernel size is too small for this differential operator.\n"
                    f"Diffop passed to RBF: {diffs} with coefficients {coefficients[nonzero]}", RuntimeWarning)
    if np.any(np.abs(out) > 15):
        warnings.warn(f"Large filter values encountered while discretizing diffop {display_diffop(coefficients)} on {num_neighbors} points. "
                    "A larger kernel size might help.\n"
                    f"Max abs filter value: {np.max(np.abs(out))}\n"
                    f"Mean abs filter value: {np.mean(np.abs(out))}\n"
                    f"Diffop passed to RBF: {diffs} with coefficients {coefficients[nonzero]}\n"
                    f"Number of output points: {out_coords.shape[0]}", RuntimeWarning)

    _RBFFD_CACHE[key] = out
    return out


def discretize_homogeneous_polynomial(
        points: Union[np.ndarray, Tuple[List[float], List[float]], List[float]],
        coefficients: np.ndarray,
        num_neighbors: int = None,
        smoothing: float = None,
        accuracy: int = None,
) -> Union[np.ndarray, sparse.COO]:
    """Discretize a homogeneous differential operator.

    Args:
        points (ndarray, tuple or list): To use RBF-FD, this has to be a
          2 x N array with N points on which to discretize.
          To use FD, this can be either a list of floats, which will be used
          as the 1D coordinates on which to discretize, or a tuple of two such
          lists, one for the x- and one for the y-axis.
          You can also use RBF-FD on a regular kernel,
          in that case you need to pass in the grid coordinates explicitly (see ``make_grid``).
        coefficients (ndarray): array with the coefficients of x^n, x^{n - 1}y, ..., y^n
          (in that order)
        out_coords (ndarray, optional): 2 x M array with output grid (i.e. points at which the
          derivative is to be found). If None, only the derivative at zero is found.
          Only supported for RBF-FD, so ``points`` needs to be an array if ``out_coords`` is not None.
        num_neighbors (int, optional): number of nearest neighbors that will be used to create
          a stencil. If no output coordinates are specified and this is None (the default),
          all input points are used for the stencil. If output coordinates are passed, this
          parameter must be set! Then each row of the output will have ``num_neighbors``
          non-zero entries.
          Note that if you set this parameter without specifying ``out_coords``, you will
          still get a dense array of shape (N, ), only with some entries zeroed out.
        accuracy (int, optional): if this is given, then the kernel will still have the
          shape defined by ``points``, but if possible the margins will be zero, so it
          is effectively a smaller kernel. The size of this smaller kernel is the smallest
          one with which the desired order of accuracy is reached.

    Returns:
        If ``out_coords`` is ``None``, ndarray of length N with weights for the N grid points.
        Otherwise, sparse matrix of shape (M, N)"""
    if isinstance(points, (list, tuple)):
        if isinstance(points, list):
            num_points = len(points) **2
            points_key = tuple(points)
        else:
            num_points = len(points[0]) * len(points[1])
            points_key = (tuple(points[0]), tuple(points[1]))

    else:
        assert isinstance(points, np.ndarray)
        assert points.shape[0] == 2
        points_key = points.tobytes()
        num_points = points.shape[1]

    targets = np.array([[0, 0]])
    if num_neighbors is None:
        # If we use a regular kernel and no number of neighbors is explicitly
        # specified, we use the entire kernel
        num_neighbors = num_points
    elif isinstance(points, (list, tuple)):
        raise ValueError("Specifying the number of neighbors is not supported for FD discretization. "
                            "If you really want to use a custom number of neighbors with a regular kernel, "
                            "pass in the grid coordinates explicitly (see make_grid) to switch to RBF-FD.")
    else:
        warnings.warn("You specified a number of neighbors without specifying output coordinates. "
                        "You will get a dense kernel, some entries will just be zero.")

    key = (points_key, coefficients.tobytes(), num_neighbors, smoothing, accuracy)
    if key in _DIFFOP_CACHE:
        return _DIFFOP_CACHE[key]

    n = len(coefficients) - 1

    nonzero = (coefficients != 0)
    nonzero_indices = [k for k in range(n + 1) if nonzero[k]]

    diffs = [[n - k, k] for k in nonzero_indices]
    if not diffs:
        # we have the zero operator
        return np.zeros(num_points)
    
    if smoothing is not None:
        if accuracy is not None:
            raise NotImplementedError("Zero-padded kernels are not yet implemented for derivatives of Gaussians")
        # use derivatives of Gaussians. In this context, we don't distinguish
        # between FD/RBF-FD, we just return the derivative of a Gaussian on
        # the given points
        if isinstance(points, list):
            points = (points, points)
        if isinstance(points, tuple):
            assert len(points) == 2
            xx, yy = np.meshgrid(points[0], points[1], indexing="ij")
            grid = np.stack([xx, yy]).reshape(2, -1)
            return gaussian_derivative(coefficients, grid, smoothing)
        
        assert isinstance(points, np.ndarray)
        return gaussian_derivative(coefficients, points, smoothing)

    # No smoothing is used, the remaining implementation depends on whether
    # we use FD or RBF-FD
    if isinstance(points, (tuple, list)):
        # If points is a list or tuple, we use a regular grid
        # and the standard FD kernels
        kernels = np.stack(
            # type checkers get confused by expanding diff, but we know
            # that it has length 2.
            [discretize_2d_monomial(*diff, points, accuracy) for diff in diffs] # type: ignore
        ).reshape(-1, num_points)

        out = np.sum(
            coefficients[nonzero][:, None] * kernels, axis=0
        )
    else:
        if accuracy is not None:
            raise ValueError("Can't use target accuracy with RBF-FD, set num_neighbors explicitly instead")
        # points is an array, so we use RBF-FD
        out = weight_matrix(targets, points.T, num_neighbors, diffs, coefficients[nonzero], phi=phi)
        if np.any(np.isnan(out.data)):
            raise Exception(f"NaNs encountered while discretizing diffop {display_diffop(coefficients)} on {num_neighbors} points.\n"
                            f"Diffop passed to RBF: {diffs} with coefficients {coefficients[nonzero]}")
        if np.all(out.data == 0):
            warnings.warn(f"Zero filter encountered while discretizing diffop {display_diffop(coefficients)} on {num_neighbors} points. "
                        "This might indicate that the kernel size is too small for this differential operator.\n"
                        f"Diffop passed to RBF: {diffs} with coefficients {coefficients[nonzero]}", RuntimeWarning)
        if np.any(np.abs(out.data) > 1e2):
            warnings.warn(f"Large filter values encountered while discretizing diffop {display_diffop(coefficients)} on {num_neighbors} points. "
                        "A larger kernel size might help.\n"
                        f"Max abs filter value: {np.max(np.abs(out))}\n"
                        f"Diffop passed to RBF: {diffs} with coefficients {coefficients[nonzero]}", RuntimeWarning)

        out = out.toarray().flatten()
    _DIFFOP_CACHE[key] = out
    return out


def discretize_1d_monomial(n: int, points: List[float], accuracy: int = None) -> np.ndarray:
    """Discretize the differential operator d^n/dx^n as a convolutional kernel."""
    # calculating the finite difference coefficients using sympy is fast,
    # but given that this function is called extremely often when sampling
    # a basis, it's still a bottleneck. Even with the cache on the level
    # of differential operators (because that one only caches something
    # if exactly the same operator appears multiple times).
    # So we use an additional cache here, which caches single monomials.
    key = (n, tuple(points), accuracy)
    if key not in _1D_KERNEL_CACHE:
        assert n < len(points), (f"Can't discretize differential operator of order {n} on {len(points)} points, "
                                f"at least {n + 1} points are needed")
        if accuracy is None:
            _1D_KERNEL_CACHE[key] = fd_weights(n, points)
        else:
            num_points = required_points(n, accuracy)
            # make number of points odd
            # HACK: this is probably not what we want in all scenarios,
            # just the ones I'm working with at the moment
            if num_points % 2 == 0:
                num_points += 1
            assert num_points <= len(points), (f"Can't achieve desired accuracy order {accuracy} "
                                               f"for differential operator of order {n}, "
                                               f"at least {num_points} points are needed")
            # we want to roughly use the middle portion of the given points,
            # because points are usually chosen symmetrically around 0
            full_size = len(points)
            offset = (full_size - num_points) // 2
            points = points[offset:offset + num_points]
            result = np.zeros(full_size)
            result[offset:offset + num_points] = fd_weights(n, points)
            _1D_KERNEL_CACHE[key] = result
    return _1D_KERNEL_CACHE[key]


def fd_weights(n, points):
        weights = finite_diff_weights(n, points, 0)
        # first -1 is because we want the highest order (n),
        # second -1 means we want the most accurate approximation (using all points)
        return np.array(weights[-1][-1], dtype=float)


def discretize_2d_monomial(n_x: int,
                           n_y: int,
                           points: Union[Tuple[List[float], List[float]], List[float]],
                           accuracy: int = None,
                           ) -> np.ndarray:
    """Discretize the differential operator d^{n_x + n_y}/{dx^n_x dy^n_y}."""
    if not isinstance(points, tuple):
        points = (points, points)
    x_kernel = discretize_1d_monomial(n_x, points[0], accuracy)
    y_kernel = discretize_1d_monomial(n_y, points[1], accuracy)
    return x_kernel[:, None] * y_kernel[None, :]


def multiply_polynomials(a: np.ndarray, b: np.ndarray):
    """Multiply two homogeneous polynomials.

    Args:
        a (ndarray): coefficients of x^n, x^{n - 1}y, ..., y^n for the first polynomial
        b (ndarray): coefficients for the second polynomial

    Returns:
        coefficients of the product
    """
    n, m = len(a), len(b)

    # compute the Cauchy product
    return np.array([
        sum(a[l] * b[k - l] for l in range(k + 1) if l < n and (k - l) < m)
        for k in range(n + m - 1)
    ])


def expand_binomial(a, b, exponent):
    """Expand (ax + by)^exponent in terms of monomials and return their coefficients."""
    ks = np.arange(exponent + 1)
    binom_coeffs = scipy.special.binom(exponent, ks)
    return binom_coeffs * a ** (exponent - ks) * b ** ks


def transform_polynomial(coefficients: np.ndarray, matrix: np.ndarray):
    """Calculate the coefficients of P(Ax), where P is the homogeneous polynomial
    defined by the given coefficients and A is the given matrix.
    
    Args:
        coefficients: ndarray of shape(..., n + 1), where the last axis enumerates
            coefficients for x^n, x^{n - 1}y, ..., y^n. The other axes are batch
            dimension.
        matrix: ndarray of shape (2, 2)"""
    n = coefficients.shape[-1] - 1
    batch_shape = coefficients.shape[:-1]
    # the transformed polynomial will have the same degree:
    transformed = np.zeros(batch_shape + (n + 1, ))
    # now we iterate over all n + 1 coefficients:
    for i in range(n + 1):
        # we have a term coeff * x^{n - i} * y^i
        # First, we calculate the transformed versions of the x and y terms:
        # (x')^{n - i} = (A_11 x + A_21 y)^{n - i}
        x_trafo = expand_binomial(matrix[0, 0], matrix[1, 0], n - i)
        # (y')^i = (A_12 x + A_22 y)^i
        y_trafo = expand_binomial(matrix[0, 1], matrix[1, 1], i)
        # then we combine them
        xy_trafo = multiply_polynomials(x_trafo, y_trafo)
        # and finally scale by the coefficient
        # broadcasting: (*batch_shape, 1) * (1, ..., 1, n + 1) -> (*batch_shape, n + 1)
        transformed += coefficients[..., i, None] * xy_trafo[(None, ) * len(batch_shape)]
    return transformed


def gaussian_derivative(coefficients: np.ndarray, points: np.ndarray, std: float = 1.):
    n = len(coefficients) - 1
    num_points = points.shape[1]

    nonzero = (coefficients != 0)
    nonzero_indices = [k for k in range(n + 1) if nonzero[k]]

    diffs = [[n - k, k] for k in nonzero_indices]
    if not diffs:
        # we have the zero operator
        return np.zeros(num_points)
    
    center = np.zeros((1, 2))

    kernels = np.stack(
        [_gaussian(points.T, center, 1/std**2, np.array(diff)).flatten() for diff in diffs]
    )

    return np.sum(
        coefficients[nonzero][:, None] * kernels, axis=0
    )



def homogenized_cheby(n: int, kind: str = "t") -> np.ndarray:
    """Compute the coefficients for the homogenized version of T_n or U'_n.

    Args:
        n: degree of the polynomial. May be negative, in that case the degree will
           be abs(n) but the sign may differ (see notes for exact definition)
        kind: Either 't' for the first kind or 'u' for the second kind

    Returns:
        ndarray with coefficients, ordered in descending order of the power of x,
        i.e. x^n, x^{n - 1}y, ..., y^n."""
    sign = np.sign(n)
    n = abs(n)

    kind = kind.lower()

    result = np.zeros(n + 1)
    if kind == "t":
        q = n // 2
        ks = np.arange(q + 1)
        # these are the coefficients of x^{n - 2k}y^{2k}, from k = 0 to n/2
        result[::2] = scipy.special.binom(n, 2 * ks) * (-1) ** ks
    elif kind == "u":
        q = (n - 1) // 2
        ks = np.arange(q + 1)
        # these are the coefficients of x^{n - 2k - 1}y^{2k + 1}, from k = 0 to (n - 1)/2
        result[1::2] = sign * scipy.special.binom(n, 2 * ks + 1) * (-1) ** ks
    else:
        raise ValueError("kind must be either 'u' or 't'")
    return result


def laplacian_power(k: int):
    """Compute the coefficients of x^n, x^{n - 1}y, ..., y^n for the k-th power of the Laplacian."""
    result = np.zeros(2*k + 1)
    # The k-th power is given by (x^2 + y^2)^k = sum_{i = 0}^k {k choose i} x^{2i}y^{2(k - i)}.
    # So the coefficient of x^{2i}y^{2(k - i)} is the binomial coefficient, and the coefficients
    # at odd indices are zero
    result[::2] = scipy.special.binom(k, np.arange(k + 1))
    return result


def display_diffop(coefficients: np.ndarray):
    """Show a homogeneous differential operator as a pretty string."""
    out = ""
    n = len(coefficients) - 1
    for k, coeff in enumerate(coefficients):
        if isinstance(coeff, float) and coeff.is_integer():
            coeff = int(coeff)
        if isinstance(coeff, float):
            coeff = round(coeff, 2)
        if coeff == 0:
            continue

        if out == "":
            if coeff == -1:
                out += "-"
            elif coeff != 1:
                out += f"{coeff}"
            if coeff in {1, -1} and n == 0:
                # constant term, we can't drop the 1
                out += "1"
        elif coeff == 1:
            out += " + "
        elif coeff == -1:
            out += " - "
        elif coeff > 0:
            out += f" + {coeff}"
            if not isinstance(coeff, int):
                out += " "
        elif coeff < 0:
            out += f" - {abs(coeff)}"
            if not isinstance(coeff, int):
                out += " "

        if n - k > 0:
            out += "x" + prettify_exponent(n - k)
            if n - k > 3 and k > 0:
                out += " "
        if k > 0:
            out += "y" + prettify_exponent(k)
    if out == "":
        out = "0"
    return out

def prettify_exponent(k):
    if k == 1:
        return ""
    if k == 2:
        return "²"
    if k == 3:
        return "³"
    return f"^{k}"


def make_grid(n):
    x = np.arange(-n, n + 1)
    # we want x to be the first axis, that's why we need the ::-1
    return np.stack(np.meshgrid(x, x)[::-1]).reshape(2, -1)


def eval_polys(coefficients, points):
    """Evaluate homogeneous polynomials on a set of points.

    Args:
        coefficients: list of ndarrays of shape (..., n + 1) with coefficients of x^n, x^{n - 1}y, ..., y^n
        points: ndarray with shape (2, N)

    Returns:
        ndarray with shape (L, ..., N) where N is the number of points and L the length of the input list
    """
    results = []
    for element in coefficients:
        dims = len(element.shape)
        nones = (dims - 1) * (None, )
        n = element.shape[-1] - 1
        ks = np.arange(n + 1).reshape(1, -1)
        xs = points[0].reshape(-1, 1)
        ys = points[1].reshape(-1, 1)
        monomials = xs**(n - ks) * ys**ks
        results.append((element[..., None, :] * monomials[nones]).sum(axis=-1))
    return np.stack(results)

def required_points(order: int, accuracy: int) -> int:
    """Compute the number of points necessary to achieve an approximation of the given accuracy.

    Important note: this function assumes that the points will arranged symmetrically
    around 0. Corollary 7 from https://arxiv.org/pdf/1102.3203.pdf then says that the
    order of accuracy may be boosted by 1 compared to the usual one, which is exploited in
    this function to return a lower number of points when possible.

    Args:
        order (int): order of the differential operator
        accuracy (int): desired accuracy (e.g. 2 for approximation to quadratic order)

    Returns:
        number of required sampling points
    """
    # The usual formula for the required number of points:
    N = order + accuracy
    # The remaining question is whether we can reduce this number by
    # 1 and still retain the desired accuracy, an effect called "boosted order of accuracy".
    # Corollary 7 from https://arxiv.org/pdf/1102.3203.pdf says that
    # boosting happens if the number of points minus the order is odd.
    # We want to check whether N - 1 still has the desired accuracy,
    # i.e. whether (N - 1) - order, or simply accuracy - 1, is odd:
    if (accuracy - 1) % 2:
        return N - 1
    return N

def largest_possible_order(num_points: int, accuracy: int) -> int:
    """Compute the largest diffop order such that the given accuracy is satisfied.

    See ``required_points`` for details."""
    order = num_points - accuracy
    assert order >= 0
    # check if the next-larger order would be boosted:
    if (num_points - (order + 1)) % 2:
        return order + 1
    return order

def guaranteed_accuracy(num_points: int, order: int) -> int:
    """Compute the accuracy that is guaranteed for the given order and number of points.

    See ``required_points`` for details."""
    accuracy = num_points - order
    assert accuracy >= 0
    # check if boosting happens:
    if accuracy % 2:
        return accuracy + 1
    return accuracy

def symmetric_points(n: int, dilation: float = 1) -> List[float]:
    # return e.g. [-1, 0, 1] for n = 3 and [-0.5, 0.5] for n = 2, etc.
    points = range(n)
    offset = (n - 1) / 2
    return [(x - offset) * dilation for x in points]

# See https://stackoverflow.com/questions/47269390/numpy-how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def max_partial_orders(coefficients: np.ndarray):
    """Compute the maximum partial derivative order of a PDO.

    Args:
        coefficients (ndarray): coefficients in the homogeneous format.
            Last axis should contain coefficients. Maximum of the derivative
            orders will be taken over previous axes.
    
    Returns:
        Tuple (max_x, max_y), where max_{direction} is the maximum power
        with which d/d{direction} occurs.
        
        Careful: these are floats because they may be negative infinity!
        (for the zero operator)"""
    order = coefficients.shape[-1] - 1
    assert order >= 0
    max_x = order - first_nonzero(coefficients, -1, invalid_val=np.inf).min()
    max_y = last_nonzero(coefficients, -1, -np.inf).max()
    return max_x, max_y

def nearest_neighbors(
    in_coords: np.ndarray,
    out_coords: np.ndarray,
    num_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create nearest neighbor stencils for a collection of input and output points.

    Args:
        in_coords: (D, N) array with coordinates of the input points (i.e. points
            that are available for the stencil)
        out_coords: (D, M) array of points around which stencils should be centered
        num_neighbors: number of elements in each stencil
    
    Returns:
        (M, num_neighbors) array of indices into in_coords and
        a (D, M, num_neighbors) array with the corresponding stencil points.
    """
    # indices will be an ndarray of shape n_out x num_neighbors,
    # where n_out is the number of out points
    _, indices = KDTree(in_coords.T).query(out_coords.T, num_neighbors)
    return indices, in_coords[:, indices]