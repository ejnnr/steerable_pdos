import itertools
import math
from typing import Tuple
import numpy as np
from scipy.signal import convolve2d, correlate2d
from e2cnn.diffops.utils import *

def polynomial_derivative(coefficients: np.ndarray, diff: Tuple[int, int]) -> np.ndarray:
    """Compute the derivative of a homogeneous polynomial.

    Args:
        coefficients (ndarray): coefficients of x^n, x^{n - 1}y, ..., y^n
        diff (tuple): (n_x, n_y), where n_x and n_y are the derivative orders
            with respect to x and y

    Returns:
        ndarray of shape (n - n_x - n_y + 1, ) with the coefficients of the
        (homogeneous) derivative"""
    n_x, n_y = diff
    n = len(coefficients) - 1
    order = n - n_x - n_y
    if order < 0:
        return np.array([0])
    out = np.empty(order + 1)
    for j in range(len(out)):
        # the j-th coefficient is that of x^{order - j}y^j
        # It comes from taking the derivative of the (j + n_y)-th term
        out[j] = (coefficients[j + n_y]
                  * np.prod(np.arange(order - j + 1, order - j + n_x + 1))
                  * np.prod(np.arange(j + 1, j + n_y + 1))
                  )
    return out


def test_derivative_helper():
    polys = [
        # format: (diffop, polynomial, derivative)
        ((1, 0), [-3, 2, 5], [-6, 2]),
        ((1, 1), [-3, 2, 5], [2]),
        ((1, 2), [2, 1, -3, 4, -1], [-12, 24]),
    ]
    for diff, input, output in polys:
        assert np.all(polynomial_derivative(input, diff) == output)


def test_1d_monomial_length_and_parity():
    for n in range(7):
        points = symmetric_points(required_points(n, 2))
        kernel = discretize_1d_monomial(n, points)
        assert len(kernel) == len(points)
        # check that kernel is even/odd
        for k in range(len(kernel)):
            assert kernel[k] == (-1)**n * kernel[-(k + 1)]


def test_exact_on_polynomials():
    for n in range(7):
        size = required_points(n, 2)
        # round up to the next odd number, otherwise we'd have
        # to differentiate between odd and even sizes in the code below
        if size % 2 == 0:
            size += 1
        points = symmetric_points(size)
        N = 20
        grid = make_grid(N)
        # the convolution will discard some boundary values, so need a smaller grid for final comparison
        small_N = N - (size // 2)
        small_grid = make_grid(small_N)
        coefficients = np.random.randn(n + 1)
        kernel = discretize_homogeneous_polynomial(points, coefficients).reshape(size, size)
        for degree in range(3):
            # generate random homogeneous polynomial
            poly = np.random.randn(degree + 1)
            poly_values = eval_polys([poly], grid).reshape(2 * N + 1, 2 * N + 1)
            # NOTE: I think scipy uses a different convention for correlations than Pytorch
            # and e.g. Wikipedia do? That means the kernel needs to come second, otherwise
            # the output will be flipepd.
            # For example, correlate([1], [1, 2, 3]) = [3, 2, 1] with scipy,
            # while correlate([1, 2, 3], [1]) = [1, 2, 3]
            discrete_result = correlate2d(poly_values, kernel, mode="valid")
            derivative = sum(
                coefficients[k] * polynomial_derivative(poly, (n - k, k))
                for k in range(n + 1)
            )
            exact_result = eval_polys([derivative], small_grid).reshape(2 * small_N + 1, 2 * small_N + 1)
            assert np.allclose(discrete_result, exact_result), (f"n: {n}, degree: {degree},\n"
                                                                f"poly: {poly},\n coefficients: {coefficients}\n"
                                                                f"kernel: \n{kernel}")
