from numpy import linspace

from .FRACTION_GRID_EXTENSION import FRACTION_GRID_EXTENSION
from .N_GRID import N_GRID


def make_vector_grid(
    vector,
    grid_min=None,
    grid_max=None,
    fraction_grid_extension=FRACTION_GRID_EXTENSION,
    n_grid=N_GRID,
):

    if grid_min is None:

        grid_min = vector.min()

    if grid_max is None:

        grid_max = vector.max()

    grid_range = grid_max - grid_min

    grid_extension = grid_range * fraction_grid_extension

    return linspace(grid_min - grid_extension, grid_max + grid_extension, num=n_grid)
