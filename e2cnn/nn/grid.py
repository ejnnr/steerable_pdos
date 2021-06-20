from typing import Collection, List, Tuple
import numpy as np
from rbf.pde.halton import halton_sequence

class Grid:
    def __init__(self, coordinates: np.ndarray, copy: bool = False):
        """A collection of points in Euclidean space on which the data lives.

        Warning: this makes the array that is passed in immutable, because
        the Grid instances themselves should be immutable. If you don't want
        that, set copy to True.

        Args:
            coordinates (ndarray): D x N array of coordinates, where D is the
              spatial dimension and N the number of points
            copy (bool, optional): whether to make a copy of the coordinates array.
              If this is False, the coordinates array is made read-only. No other
              changes are made to the array. Default is False."""
        assert isinstance(coordinates, np.ndarray)
        assert len(coordinates.shape) == 2, coordinates.shape
        if copy:
            coordinates = coordinates.copy()

        # We want Grid objects to be completely immutable because they should
        # be hashable and usable as dictionary keys
        coordinates.setflags(write=False)
        self._coordinates = coordinates

    @property
    def is_integer(self) -> bool:
        """Whether the grid consists only of integer coordinates."""
        return np.issubdtype(self.coordinates.dtype, np.integer)

    @classmethod
    def regular(cls, size: Collection[int], center: np.ndarray = None, dilation: float = 1, integer: bool = True) -> "Grid":
        """Create a regular grid with a given size and center.

        Args:
            size (collection of ints): sizes of the grid in each dimension. The dimension
              of the grid will be the length of this collection. Can be Tuple, List, ndarray,
              or any other type that implements ``abc.collections.Collection``.
            center (ndarray, optional): coordinates of the grid center. If not given,
              the grid will be centered around the origin. Note that the center of the grid
              is not necessarily itself a grid point (it only is if the size is odd along
              all dimensions).
            dilation (float, optional): distance between grid points. Default is 1.
            integer (bool, optional): whether to cast the grid to integer dtype if possible.
              Default is True, meaning that if all grid coordinates are integers, an integer
              dtype will be used. This means the grid coordinates can be used for indexing.
        """
        coords = _regular_grid_coords(size, dilation)
        if center is not None:
            coords += center[:, None]

        # check if all entries are integers
        if integer and np.all(np.mod(coords, 1) == 0):
            coords = coords.astype(int)

        return cls(coords, copy=False)

    def subgrid(self, indices: np.ndarray):
        """Create a subgrid by only using certain grid points.

        Args:
            indices (ndarray): indices of the grid points that should be used
        """
        coords = self.coordinates[:, indices]
        return Grid(coords)

    def random_subgrid(self, num_points: int):
        """Create a random subgrid.
        """
        indices = np.random.choice(len(self), num_points, replace=False)
        return self.subgrid(indices)

    def pooled(self, factor: float, ranges: List[Tuple[float, float]] = None):
        """Create a coarser grid with fewer points.

        Args:
            factor (float): factor by which to (approximately) reduce the number of grid points.
        """
        if ranges is None:
            ranges = []
            for i in range(self.dimension):
                coords = self.coordinates[i]
                ranges.append((np.min(coords), np.max(coords)))
        num_points = round(len(self) / factor)
        min_vals, max_vals = zip(*ranges)
        min_vals = np.array(list(min_vals)).reshape(-1, 1)
        max_vals = np.array(list(max_vals)).reshape(-1, 1)
        coords = np.random.uniform(min_vals, max_vals, (self.dimension, num_points))
        return Grid(coords)
    
    def halton(self, factor: float, integers: bool = False, ranges: List[Tuple[float, float]] = None):
        """Create a coarser grid with fewer points using a Halton sequence.

        Args:
            factor (float): factor by which to (approximately) reduce the number of grid points.
        """
        if ranges is None:
            ranges = []
            for i in range(self.dimension):
                coords = self.coordinates[i]
                ranges.append((np.min(coords), np.max(coords)))
        num_points = round(len(self) / factor)
        min_vals, max_vals = zip(*ranges)
        min_vals = np.array(list(min_vals)).reshape(-1, 1)
        max_vals = np.array(list(max_vals)).reshape(-1, 1)
        halton = halton_sequence(num_points, self.dimension).T
        halton *= max_vals - min_vals
        halton += min_vals
        if integers:
            halton = np.around(halton).astype(int)
        return Grid(halton)
    
    def plot(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.coordinates[0], self.coordinates[1])

    @property
    def coordinates(self) -> np.ndarray:
        """Coordinates of the grid points.
        This is a D x N ndarray, where D is the spatial dimension and N the number of points.
        """
        return self._coordinates

    def __len__(self) -> int:
        """The number of points in the grid."""
        return self.coordinates.shape[1]

    @property
    def dimension(self) -> int:
        """The spatial dimension of the grid."""
        return self.coordinates.shape[0]

    def __iter__(self):
        """Return an iterator over the list of points.

        Note: for large grids, use the ``Grid.coordinates`` property
        instead and use numpy operations if possible to improve performance."""

        # coordinates are dimension x num_points, but we want to iterate
        # over points, so transpose first
        return iter(self.coordinates.T)

    def __eq__(self, other):
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self.coordinates, other.coordinates)

    def __hash__(self):
        return hash(self.coordinates.tobytes())

    def __repr__(self):
        return f"Grid(dim={self.dimension}, num_points={len(self)}, coordinates={self.coordinates!r})"

    def __str__(self):
        return f"<Grid with {self.dimension} dimensions and {len(self)} points>"


def _regular_grid_coords(size, dilation):
    axes = []
    for dim in size:
        half_dim = (dim - 1) / 2
        # for e.g. dim = 5, we want [-2, -1, 0, 1, 2], for
        # dim = 4, we want [-1.5, 0.5, 0.5, 1.5]
        # and for dilations != 1, we want to scale each of these values.
        # That's what this line does
        axes.append(np.arange(-half_dim, half_dim + 1) * dilation)
    # we want x to be the first axis, that's why we need the ::-1
    return np.stack(np.meshgrid(*axes)[::-1]).reshape(len(size), -1)
