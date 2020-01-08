from numpy import linspace


def make_grid(
    grid_min, grid_max, fraction_grid_extension, n_grid,
):

    if 0 < fraction_grid_extension:

        grid_extension = (grid_max - grid_min) * fraction_grid_extension

        grid_min -= grid_extension

        grid_max += grid_extension

    return linspace(grid_min, grid_max, num=n_grid)
