import numpy as np
import pytest

from e2cnn.nn import Grid

def test_create_regular():
    grid = Grid.regular((5, ))
    assert grid.dimension == 1
    assert len(grid) == 5
    assert np.array_equal(grid.coordinates, np.array([[-2, -1, 0, 1, 2]]))
    grid = Grid.regular((4, ))
    assert grid.dimension == 1
    assert len(grid) == 4
    assert np.array_equal(grid.coordinates, np.array([[-1.5, -0.5, 0.5, 1.5]]))

    grid = Grid.regular((4, 5, 6))
    assert grid.dimension == 3
    assert len(grid) == 4 * 5 * 6

    center = np.array([1, 2, 3])
    shifted_grid = Grid.regular((4, 5, 6), center=center)
    assert np.array_equal(grid.coordinates + center[:, None], shifted_grid.coordinates)

    dilated_grid = Grid.regular((4, 5, 6), center=center, dilation=2)
    assert np.array_equal(2 * grid.coordinates + center[:, None], dilated_grid.coordinates)

def test_iter():
    coords = np.arange(10).reshape(2, 5)
    grid = Grid(coords)
    for i, point in enumerate(grid):
        assert isinstance(point, np.ndarray)
        assert np.array_equal(point, coords[:, i])

def test_read_only():
    coords = np.arange(10).reshape(2, 5)
    grid = Grid(coords, copy=True)
    with pytest.raises(AttributeError):
        grid.coordinates = np.arange(5)
    with pytest.raises(ValueError):
        grid.coordinates[0, 0] = 1

    grid = Grid(coords, copy=False)
    with pytest.raises(ValueError):
        grid.coordinates[0, 0] = 1
    with pytest.raises(ValueError):
        coords[0, 0] = 1

def test_hash_and_equality():
    coords = np.arange(10).reshape(2, 5)
    grid = Grid(coords)

    coords = np.arange(10).reshape(2, 5)
    grid2 = Grid(coords)

    assert grid == grid2
    assert hash(grid) == hash(grid2)

    coords = coords + 1
    grid2 = Grid(coords)
    assert grid != grid2
    assert hash(grid) != hash(grid2)

def test_integer():
    center = np.array([1., 2., 3.])
    grid = Grid.regular((4, 5, 6), center=center)
    assert not grid.is_integer
    grid = Grid.regular((3, 5, 7), center=center)
    assert grid.is_integer
    grid = Grid.regular((3, 5, 7), center=center, dilation=2.0)
    assert grid.is_integer
    grid = Grid.regular((3, 5, 7), center=center, dilation=2.1)
    assert not grid.is_integer
    center += 0.1
    grid = Grid.regular((3, 5, 7), center=center, dilation=2)
    assert not grid.is_integer
