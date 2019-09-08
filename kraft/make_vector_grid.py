from numpy import linspace

from .FRACTION_GRID_EXTENSION_FOR_ESTIMATING_KERNEL_DENSITY import (
    FRACTION_GRID_EXTENSION_FOR_ESTIMATING_KERNEL_DENSITY,
)
from .N_GRID_FOR_ESTIMATING_KERNEL_DENSITY import N_GRID_FOR_ESTIMATING_KERNEL_DENSITY


def make_vector_grid(
    vector,
    grid_min=None,
    grid_max=None,
    fraction_grid_extension=FRACTION_GRID_EXTENSION_FOR_ESTIMATING_KERNEL_DENSITY,
    n_grid=N_GRID_FOR_ESTIMATING_KERNEL_DENSITY,
):

    if grid_min is None:

        grid_min = vector.min()

    if grid_max is None:

        grid_max = vector.max()

    grid_range = grid_max - grid_min

    grid_extension = grid_range * fraction_grid_extension

    return linspace(grid_min - grid_extension, grid_max + grid_extension, num=n_grid)
