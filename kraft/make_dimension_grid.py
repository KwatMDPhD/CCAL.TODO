from numpy import linspace


def make_dimension_grid(
    grid_min, grid_max, fraction_grid_extension, n_grid,
):

    if 0 < fraction_grid_extension:

        grid_extension = fraction_grid_extension * (grid_max - grid_min)

        grid_min -= grid_extension

        grid_max += grid_extension

    return linspace(grid_min, grid_max, num=n_grid)
