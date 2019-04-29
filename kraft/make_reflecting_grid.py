from .check_nd_array_for_bad import check_nd_array_for_bad


def make_reflecting_grid(grid, reflecting_grid_value, raise_for_bad=True):

    check_nd_array_for_bad(grid, raise_for_bad=raise_for_bad)

    reflecting_grid = grid.copy()

    for i, grid_value in enumerate(reflecting_grid):

        if grid_value < reflecting_grid_value:

            reflecting_grid[i] += (reflecting_grid_value - grid_value) * 2

        else:

            reflecting_grid[i] -= (grid_value - reflecting_grid_value) * 2

    return reflecting_grid
