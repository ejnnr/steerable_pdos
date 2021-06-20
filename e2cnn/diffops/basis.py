
import itertools
from e2cnn.kernels.basis import EmptyBasisException
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Union, Tuple

from e2cnn.kernels import KernelBasis
from e2cnn.group import SO2
from e2cnn.nn import HybridTensor
from .utils import discretize_homogeneous_polynomial, max_partial_orders, multiply_polynomials, laplacian_power, display_diffop, rbffd_discretization, nearest_neighbors, transform_polynomial


class DiffopBasis(KernelBasis):

    def __init__(self, coefficients: List[np.ndarray]):
        r"""
        Abstract class for implementing the basis of a space of differential operators.
        Such a space consists of :math:`c_\text{out} \times c_\text{in}` matrices with
        partial differential operators as entries.
        
        Args:
            coefficients (list): a list of ndarrays. Each array describes one element
                of the basis and has shape ``(c_out, c_in, n + 1)``, where ``n``
                is the derivative order of the entries of the matrix.
                PDOs are encoded as the coefficients of :math:`\frac{\partial^n}{\partial x^n}`,
                :math:`\frac{\partial^n}{\partial x^{n - 1}\partial y}`, ...,
                :math:`\frac{\partial^n}{\partial y^n}`.
            
        Attributes:
            ~.coefficients (list): an analytical description of the PDO basis elements, see above
            ~.dim (int): the dimensionality of the basis :math:`|\mathcal{K}|` (number of elements)
            ~.shape (tuple): a tuple containing :math:`c_\text{out}` and :math:`c_\text{in}`

        """
        dim = len(coefficients)
        if dim == 0:
            raise EmptyBasisException
        shape = coefficients[0].shape[:2]
        self.maximum_order = 0
        for element in coefficients:
            assert element.shape[:2] == shape
            assert len(element.shape) == 3
            # we sometimes get very small coefficients (on the order of 1e-17)
            # through rounding errors, those should be 0
            # this is important to get the derivative orders right for basis filters
            element[np.abs(element) < 1e-8] = 0
            # We want to know the maximum order that appears in this basis.
            # The last axis contains the actual derivative, and has length order + 1
            self.maximum_order = max(self.maximum_order, element.shape[-1] - 1)
        self.coefficients = coefficients
        super().__init__(dim, shape)

    def sample(self,
             points: Union[np.ndarray, List[float], Tuple[List[float], List[float]]],
             out_coords: np.ndarray = None,
             mask: np.ndarray = None,
             num_neighbors: int = None,
             smoothing: float = None,
             accuracy: int = None,
             angle_offset: float = None,
             ) -> Union[np.ndarray, HybridTensor]:
        """Discretize the basis on a set of points.

        Args:
            points (ndarray, tuple or list): To use RBF-FD, this has to be a
              2 x N array with N points on which to discretize.
              To use FD, this can be either a list of floats, which will be used
              as the 1D coordinates on which to discretize, or a tuple of two such
              lists, one for the x- and one for the y-axis.
              You can also use RBF-FD on a regular kernel,
              in that case you need to pass in the grid coordinates explicitly (see ``make_grid``).
            out (ndarray, optional): array into which to write the output. Only supported if no custom
              ``out_coords`` are set.
            mask (ndarray, optional): Boolean array of shape (dim, ), where ``dim`` is the number of basis elements.
              True for elements to discretize and False for elements to discard.
            out_coords (ndarray, optional): 2 x M array with output grid (i.e. points at which the
              derivative is to be found). If None, only the derivative at zero is found.
              Only supported for RBF-FD, so ``points`` needs to be an array if ``out_coords`` is not None.
            num_neighbors (int, optional): number of nearest neighbors that will be used to create
              a stencil. If no output coordinates are specified and this is None (the default),
              all input points are used for the stencil. If output coordinates are passed, this
              parameter must be set! Then each row of the output will have ``num_neighbors``
              non-zero entries.
              Note that if you set this parameter without specifying ``out_coords``, you will
              still get a dense array, only with some entries zeroed out.

        Returns:
            If ``out_coords`` is ``None``, ndarray of with shape (C_out, C_in, num_basis_elements, n_in), where
            ``num_basis_elements`` are the number of elements after applying the mask, and ``n_in`` is the number
            of points.
            Otherwise, a sparse.COO array of shape (c_out, c_in, num_basis_elements, n_out, n_in), where n_out
            and n_in are the number of output and input points.
        """
        if mask is None:
            # if no mask is given, we use all basis elements
            mask = np.array([True] * self.dim)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (self.dim, )

        if angle_offset is not None:
            so2 = SO2(1)
            # rotation matrix by angle_offset
            matrix = so2.irrep(1)(angle_offset)
            # we transform the polynomial with the matrix
            coefficients = (transform_polynomial(element, matrix) for element in self.coefficients)
        else:
            coefficients = self.coefficients

        coefficients = (coeff for coeff, m in zip(coefficients, mask) if m)

        if out_coords is None:
            return self._sample_dense(coefficients, points, mask, num_neighbors, smoothing, accuracy)
        else:
            assert isinstance(points, np.ndarray)
            assert isinstance(out_coords, np.ndarray)
            assert isinstance(num_neighbors, int)
            return self._sample_sparse(coefficients, points, out_coords, mask, num_neighbors)

    def pretty_print(self):
        out = ""
        for element in self.coefficients:
            out += display_matrix(element)
            out += "\n----------------------------------\n"
        return out
    
    def _sample_sparse(
        self,
        coefficients: Iterable[np.ndarray],
        points: np.ndarray,
        out_coords: np.ndarray,
        mask: np.ndarray,
        num_neighbors: int,
    ) -> HybridTensor:
        values = np.empty((np.sum(mask), ) + self.shape + (out_coords.shape[1], num_neighbors), dtype=np.float32)

        indices, stencils = nearest_neighbors(points, out_coords, num_neighbors)

        for k, element in enumerate(coefficients):
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    values[k, i, j] = rbffd_discretization(element[i, j], out_coords, stencils)
        
        return HybridTensor(torch.from_numpy(indices).long(), torch.from_numpy(values), points.shape[1])
    
    def _sample_dense(
        self,
        coefficients: Iterable[np.ndarray],
        points: Union[np.ndarray, List[float], Tuple[List[float], List[float]]],
        mask: np.ndarray,
        num_neighbors: Optional[int],
        smoothing: Optional[float],
        accuracy: Optional[int],
    ) -> np.ndarray:
        if isinstance(points, np.ndarray):
            assert len(points.shape) == 2
            assert points.shape[0] == 2
            num_points = points.shape[1]
        elif isinstance(points, list):
            num_points = len(points) ** 2
            for x in points:
                assert isinstance(x, float)
        else:
            assert isinstance(points, tuple)
            assert len(points) == 2
            num_points = len(points[0]) * len(points[1])
            for i in range(2):
                assert isinstance(points[i], list)
                for x in points[i]:
                    assert isinstance(x, float)

        basis = np.empty((np.sum(mask), ) + self.shape + (num_points, ))
        for k, element in enumerate(coefficients):
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    basis[k, i, j] = discretize_homogeneous_polynomial(points, element[i, j], num_neighbors, smoothing, accuracy)

        # Finally, we move the len_basis axis to the third position
        basis = basis.transpose(1, 2, 0, 3)

        return basis


class LaplaceProfile(DiffopBasis):

    def __init__(self, max_power: int):
        r"""
        Basis for rotationally invariant PDOs.
        Each basis element is defined as a power of a Laplacian.

        In order to build a complete basis of PDOs, you should combine this basis
        with a basis which defines the angular profile through :class:`~e2cnn.diffops.TensorBasis`.


        Args:
            max_power (int): the maximum power of the Laplace operator that will be used.
                The maximum degree (as a differential operator) will be two times this maximum
                power.

        """

        assert isinstance(max_power, int)
        assert max_power >= 0

        coefficients = [
            laplacian_power(k).reshape(1, 1, -1) for k in range(max_power + 1)
        ]

        super().__init__(coefficients)

        self.max_power = max_power

    def __getitem__(self, r):
        assert r < self.dim
        return {"power": r, "order": 2 * r, "idx": r}

    def __eq__(self, other):
        if isinstance(other, LaplaceProfile):
            return self.max_power == other.max_power
        else:
            return False

    def __hash__(self):
        return hash(self.max_power)


class TensorBasis(DiffopBasis):

    def __init__(self, basis1: DiffopBasis, basis2: DiffopBasis):
        r"""

        Build the tensor product basis of two PDO bases over the
        plane. Given two bases :math:`A = \{a_i\}_i` and :math:`B = \{b_j\}_j`, this basis is defined as

        .. math::
            C = A \otimes B = \left\{ c_{i,j} := a_i \circ b_j \right\}_{i,j}.

        Args:
            basis1 (DiffopBasis): the first basis
            basis2 (DiffopBasis): the second basis

        Attributes:
            ~.basis1 (DiffopBasis): the first basis
            ~.basis2 (DiffopBasis): the second basis

        """
        coefficients = []
        for a, b in itertools.product(basis1.coefficients, basis2.coefficients):
            order = a.shape[2] + b.shape[2] - 2
            out = np.empty((a.shape[0], b.shape[0], a.shape[1], b.shape[1], order + 1))
            for i, j, k, l in itertools.product(range(a.shape[0]),
                                             range(b.shape[0]),
                                             range(a.shape[1]),
                                             range(b.shape[1])):
                out[i, j, k, l] = multiply_polynomials(a[i, k], b[j, l])
            out = out.reshape(a.shape[0] * b.shape[0], a.shape[1] * b.shape[1], order + 1)
            coefficients.append(out)
        super().__init__(coefficients)
        self.basis1 = basis1
        self.basis2 = basis2

    def __getitem__(self, idx):
        assert idx < self.dim
        idx1, idx2 = divmod(idx, self.basis2.dim)
        attr1 = self.basis1[idx1]
        attr2 = self.basis2[idx2]

        attr = dict()
        attr.update(attr1)
        attr.update(attr2)

        attr["order"] = attr1["order"] + attr2["order"]

        attr["idx"] = idx
        attr["idx1"] = idx1
        attr["idx2"] = idx2

        attr["max_x"], attr["max_y"] = max_partial_orders(self.coefficients[idx][0])

        return attr

    def __iter__(self):
        idx = 0
        for attr1 in self.basis1:
            for attr2 in self.basis2:
                attr = dict()
                attr.update(attr1)
                attr.update(attr2)
                attr["order"] = attr1["order"] + attr2["order"]
                attr["idx"] = idx
                attr["idx1"] = attr1["idx"]
                attr["idx2"] = attr2["idx"]

                attr["max_x"], attr["max_y"] = max_partial_orders(self.coefficients[idx][0])

                yield attr
                idx += 1

    def __eq__(self, other):
        if isinstance(other, TensorBasis):
            return self.basis1 == other.basis1 and self.basis2 == other.basis2
        else:
            return False

    def __hash__(self):
        return hash(self.basis1) + hash(self.basis2)


def display_matrix(element):
    out = ""
    for i in range(element.shape[0]):
        for j in range(element.shape[1]):
            out += display_diffop(element[i, j]) + "\t"
        out += "\n"
    return out
