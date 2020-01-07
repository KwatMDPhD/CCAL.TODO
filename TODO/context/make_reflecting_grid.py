from .is_array_bad import is_array_bad


def make_reflecting_grid(grid, reflecting_grid_value, raise_if_bad=True):

    is_array_bad(grid, raise_if_bad=raise_if_bad)

    grid_ = grid.copy()

    for i, grid_value in enumerate(grid_):

        if grid_value < reflecting_grid_value:

            grid_[i] += (reflecting_grid_value - grid_value) * 2

        else:

            grid_[i] -= (grid_value - reflecting_grid_value) * 2

    return grid_
