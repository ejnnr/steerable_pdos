
import numpy as np

from e2cnn.kernels.basis import KernelBasis, EmptyBasisException
from e2cnn.kernels.irreps_basis import IrrepBasis

from e2cnn.group import Representation, SO2, CyclicGroup

from .basis import DiffopBasis
from .utils import max_partial_orders, transform_polynomial

from typing import Type, List


class SteerableKernelBasis(DiffopBasis):

    def __init__(self,
                 irreps_basis: Type[DiffopBasis],
                 in_repr: Representation,
                 out_repr: Representation,
                 special_regular_basis: bool = False,
                 maximum_partial_order: int = None,
                 **kwargs):
        r"""
        
        Implements a general basis for the vector space of equivariant PDOs.
        A :math:`G`-equivariant PDO :math:`D(P)` for a matrix of polynomials :math:`P`, mapping between an input field, transforming under
        :math:`\rho_\text{in}` (``in_repr``), and an output field, transforming under  :math:`\rho_\text{out}`
        (``out_repr``), satisfies the following constraint:
        
        .. math ::
            
            P(gx) = \rho_\text{out}(g) P(x) \rho_\text{in}(g)^{-1} \qquad \forall g \in G, \forall x \in X
        
        for :math:`G \leq \O{d}`.

        As the PDO constraint is a linear constraint, the space of equivariant PDOs is a vector subspace of the
        space of all PDOs. It follows that any equivariant PDO can be expressed in terms of a basis
        of this space.
        
        This class solves the PDO constraint for two arbitrary representations by combining the solutions of the
        PDO constraints associated to their :class:`~e2cnn.group.IrreducibleRepresentation` s.
        In order to do so, it relies on ``irreps_basis`` which solves individual irreps constraints. ``irreps_basis``
        must be a class which builds a basis for equivariant
        kernels associated with irreducible representations when instantiated.
        
        The groups :math:`G` which are currently implemented are origin-preserving isometries (what are called
        structure groups, or sometimes gauge groups, in the language of
        `Gauge Equivariant CNNs <https://arxiv.org/abs/1902.04615>`_ ).
        The origin-preserving isometries of :math:`\R^d` are subgroups of :math:`O(d)`, i.e. reflections and rotations.
        Therefore, PDOs may be composed with any rotation and reflection invariant PDO without
        affecting equivariance. This class only implements a basis up to such invariant PDOs,
        which are given by polynomials in the Laplacian operator.
        
        In order to build a complete basis of PDOs, you should combine this basis with
        :class:`~e2cnn.diffops.LaplaceProfile`) through
        :class:`~e2cnn.diffops.TensorBasis`.
        
        .. warning ::
            
            Typically, the user does not need to manually instantiate this class.
            Instead, we suggest to use the interface provided in :doc:`e2cnn.gspaces`.
        
        Args:
            irreps_basis (class): class defining the irreps basis. This class is instantiated for each pair of irreps to solve all irreps constraints.
            in_repr (Representation): Representation associated with the input feature field
            out_repr (Representation): Representation associated with the output feature field
            **kwargs: additional arguments used when instantiating ``irreps_basis``
            
        """
        
        assert in_repr.group == out_repr.group
        
        self.in_repr = in_repr
        self.out_repr = out_repr
        group = in_repr.group
        self.group = group
        self.special_regular_basis = False

        # To implement the way that PDO-eConvs filter the basis, we need
        # a specific basis because the PDO-eConvs filter is not invariant
        # under a change of basis
        if (special_regular_basis
            and {in_repr.name, out_repr.name} in ({"regular"}, {"regular", "irrep_0"})):
            
            self.special_regular_basis = True
            # for now, this is only implemented in a very specific setting
            # that we need to implement PDO-eConvs. Should be easy to generalize
            # to D_N and to non-zero maximum offsets though
            assert isinstance(in_repr.group, CyclicGroup)
            assert "max_frequency" in kwargs
            assert "max_offset" in kwargs and kwargs["max_offset"] == 0
            if {in_repr.name, out_repr.name} == {"regular"}:
                coefficients = build_regular_basis(in_repr.group.order(), kwargs["max_frequency"], maximum_partial_order)
            elif in_repr.name == "regular":
                coefficients = build_regular_to_trivial_basis(in_repr.group.order(), kwargs["max_frequency"], maximum_partial_order)
            else:
                coefficients = build_trivial_to_regular_basis(in_repr.group.order(), kwargs["max_frequency"], maximum_partial_order)

            super().__init__(coefficients)
            return

        assert maximum_partial_order is None, "partial order limit only supported for special regular basis"

        A_inv = np.array(in_repr.change_of_basis_inv, copy=True)
        B = np.array(out_repr.change_of_basis, copy=True)
        
        # A_inv = in_repr.change_of_basis_inv
        # B = out_repr.change_of_basis

        if not np.allclose(A_inv, np.eye(in_repr.size)):
            self.A_inv = A_inv
        else:
            self.A_inv = None
            
        if not np.allclose(B, np.eye(out_repr.size)):
            self.B = B
        else:
            self.B = None

        self.irreps_bases = {}

        # loop over all input irreps
        for i_irrep_name in set(in_repr.irreps):
            # loop over all output irreps
            for o_irrep_name in set(out_repr.irreps):
        
                try:
                    # retrieve the irrep intertwiner basis
                    basis = irreps_basis(group=group,
                                         in_irrep=i_irrep_name,
                                         out_irrep=o_irrep_name,
                                         **kwargs)

                    self.irreps_bases[(i_irrep_name, o_irrep_name)] = basis

                except EmptyBasisException:
                    # if the basis is empty, skip it
                    pass
        
        self.bases = [[None for _ in range(len(out_repr.irreps))] for _ in range(len(in_repr.irreps))]
        
        self.in_sizes = []
        self.out_sizes = []
        # loop over all input irreps
        for ii, i_irrep_name in enumerate(in_repr.irreps):
            self.in_sizes.append(group.irreps[i_irrep_name].size)
            
        # loop over all output irreps
        for oo, o_irrep_name in enumerate(out_repr.irreps):
            self.out_sizes.append(group.irreps[o_irrep_name].size)

        dim = 0
        # loop over all input irreps
        for ii, i_irrep_name in enumerate(in_repr.irreps):
            # loop over all output irreps
            for oo, o_irrep_name in enumerate(out_repr.irreps):
                if (i_irrep_name, o_irrep_name) in self.irreps_bases:
                    self.bases[ii][oo] = self.irreps_bases[(i_irrep_name, o_irrep_name)]
                    dim += self.bases[ii][oo].dim

        # would be set later anyway but we need it now
        self.shape = (out_repr.size, in_repr.size)

        if self.A_inv is None and self.B is None:
            coefficients = self._direct_sum_coefficients()
        else:
            pre_coefficients = self._direct_sum_coefficients()
            coefficients = self._change_of_basis(pre_coefficients)
        
        super().__init__(coefficients)

    def _direct_sum_coefficients(self) -> List[np.ndarray]:
        coefficients: List[np.ndarray] = []
        basis_count = 0
        in_position = 0
        for ii, in_size in enumerate(self.in_sizes):
            out_position = 0
            for oo, out_size in enumerate(self.out_sizes):
                if self.bases[ii][oo] is not None:
                    block_coefficients = self.bases[ii][oo].coefficients
                    for element in block_coefficients:
                        out = np.zeros((self.shape[0], self.shape[1], element.shape[2]))
                        out[
                            out_position:out_position+out_size,
                            in_position:in_position+in_size,
                            :
                        ] = element
                        coefficients.append(out)

                out_position += out_size
            in_position += in_size

        return coefficients
    
    def _change_of_basis(self, coefficients: List[np.ndarray]) -> List[np.ndarray]:
        # multiply by the change of basis matrices to transform the irreps basis in the full representations basis
        new_coefficients: List[np.ndarray] = []
        for element in coefficients:
            if self.A_inv is not None and self.B is not None:
                new_coefficients.append(np.einsum("no,oib,ij->njb", self.B, element, self.A_inv))
            elif self.A_inv is not None:
                new_coefficients.append(np.einsum("oib,ij->ojb", element, self.A_inv))
            elif self.B is not None:
                new_coefficients.append(np.einsum("no,oib->nib", self.B, element))
            else:
                new_coefficients.append(element)

        return new_coefficients

    def __getitem__(self, idx):
        assert idx < self.dim

        if self.special_regular_basis:
            return {
                "idx": idx,
                "inner_idx": idx,
                "shape": self.coefficients[idx].shape[:2],
                "order": self.coefficients[idx].shape[2] - 1,
                "frequency": self.coefficients[idx].shape[2] - 1
            }
        
        count = 0
        for ii in range(len(self.in_sizes)):
            for oo in range(len(self.out_sizes)):
                if self.bases[ii][oo] is not None:
                    dim = self.bases[ii][oo].dim

                    rel_idx = idx - count
                    if rel_idx >= 0 and rel_idx < dim:
                        
                        attr = dict(self.bases[ii][oo][rel_idx])
                        
                        attr["shape"] = self.bases[ii][oo].shape
                        
                        attr["in_irrep"] = self.in_repr.irreps[ii]
                        attr["out_irrep"] = self.out_repr.irreps[oo]
                        
                        attr["in_irrep_idx"] = ii
                        attr["out_irrep_idx"] = oo
                        
                        attr["inner_idx"] = attr["idx"]
                        attr["idx"] = idx
                        
                        return attr
                
                    count += dim

    def __iter__(self):
        if self.special_regular_basis:
            for idx in range(len(self.coefficients)):
                yield {
                    "idx": idx,
                    "inner_idx": idx,
                    "shape": self.coefficients[idx].shape[:2],
                    "order": self.coefficients[idx].shape[2] - 1,
                    "frequency": self.coefficients[idx].shape[2] - 1
                }
            return

        idx = 0
        for ii in range(len(self.in_sizes)):
            for oo in range(len(self.out_sizes)):
                if self.bases[ii][oo] is not None:
                    for rel_idx in range(len(self.bases[ii][oo])):
                        attr = dict(self.bases[ii][oo][rel_idx])

                        attr["shape"] = self.bases[ii][oo].shape
                        
                        attr["in_irrep"] = self.in_repr.irreps[ii]
                        attr["out_irrep"] = self.out_repr.irreps[oo]
                    
                        attr["in_irrep_idx"] = ii
                        attr["out_irrep_idx"] = oo
                        
                        attr["inner_idx"] = attr["idx"]
                        attr["idx"] = idx

                        yield attr
                        
                        idx += 1

    def __eq__(self, other):
        if not isinstance(other, SteerableKernelBasis):
            return False
        elif self.in_repr != other.in_repr or self.out_repr != other.out_repr:
            return False
        elif self.special_regular_basis:
            if not other.special_regular_basis:
                return False
            # checking whether the bases are equally large, which works
            # because the only variable is the maximum frequency, which determines
            # the basis size
            return len(self.coefficients) == len(other.coefficients)
        else:
            sbk1 = sorted(self.irreps_bases.keys())
            sbk2 = sorted(other.irreps_bases.keys())
            if sbk1 != sbk2:
                return False
            
            for irreps, basis in self.irreps_bases.items():
                if basis != other.irreps_bases[irreps]:
                    return False
            
            return True

    def __hash__(self):
        key = (self.in_repr, self.out_repr, self.special_regular_basis)
        if self.special_regular_basis:
            return hash((key, len(self.coefficients)))

        h = hash(key)
        for basis in self.irreps_bases.items():
            h += hash(basis)
        return h
        
        
def build_regular_basis(N: int, maximum_order: int, maximum_partial_order: int = None):
    """Compute the coefficient list for a C_N -> C_N regular basis."""
    so2 = SO2(maximum_order)
    basis = []
    # iterate over all N freely choosable positions
    for pos in range(N):
        # then iterate over all PDO orders
        for order in range(maximum_order + 1):
            # and for each order over all monomials with that order,
            # x^order, ..., y^order
            for i in range(order + 1):
                element = np.zeros((N, N, order + 1))
                poly = np.zeros(order + 1)
                poly[i] = 1

                if maximum_partial_order is not None and any(order > maximum_partial_order for order in max_partial_orders(poly)):
                    continue

                # fill in the rows of the current basis element,
                # they are rotated and shifted versions of the first one
                for j in range(N):
                    # the rotation matrix for the j-th element of C_N:
                    matrix = so2.irrep(1)(2 * np.pi / N * j)
                    # we transform the polynomial with the matrix
                    transformed_poly = transform_polynomial(poly, matrix)
                    # then we set the right element of the matrix to that polynomial.
                    # we are in row j, and in the first row, we want the polynomial
                    # at position pos. All other rows are shifted cyclically.
                    element[j, (pos + j) % N] = transformed_poly
                basis.append(element)
    return basis

def build_trivial_to_regular_basis(N: int, maximum_order: int, maximum_partial_order: int = None):
    so2 = SO2(maximum_order)
    basis = []
    # iterate over all PDO orders
    for order in range(maximum_order + 1):
        # and for each order over all monomials with that order,
        # x^order, ..., y^order
        for i in range(order + 1):
            element = np.zeros((N, 1, order + 1))
            poly = np.zeros(order + 1)
            poly[i] = 1

            if maximum_partial_order is not None and any(order > maximum_partial_order for order in max_partial_orders(poly)):
                continue

            # fill in the rows of the current basis element,
            # they are rotated and shifted versions of the first one
            for j in range(N):
                # the rotation matrix for the j-th element of C_N:
                matrix = so2.irrep(1)(2 * np.pi / N * j)
                # we transform the polynomial with the matrix
                transformed_poly = transform_polynomial(poly, matrix)
                element[j, 0] = transformed_poly
            basis.append(element)
    return basis

def build_regular_to_trivial_basis(N: int, maximum_order: int, maximum_partial_order: int = None):
    so2 = SO2(maximum_order)
    basis = []
    # iterate over all PDO orders
    for order in range(maximum_order + 1):
        # and for each order over all monomials with that order,
        # x^order, ..., y^order
        for i in range(order + 1):
            element = np.zeros((1, N, order + 1))
            poly = np.zeros(order + 1)
            poly[i] = 1

            if maximum_partial_order is not None and any(order > maximum_partial_order for order in max_partial_orders(poly)):
                continue

            # fill in the columns of the current basis element,
            # they are rotated and shifted versions of the first one
            for j in range(N):
                # the rotation matrix for the j-th element of C_N:
                matrix = so2.irrep(1)(2 * np.pi / N * j)
                # we transform the polynomial with the matrix
                transformed_poly = transform_polynomial(poly, matrix)
                element[0, j] = transformed_poly
            basis.append(element)
    return basis
