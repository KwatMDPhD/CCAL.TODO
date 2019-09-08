from .check_array_for_bad import check_array_for_bad


def make_reflecting_grid(grid, reflecting_grid_value, raise_for_bad=True):

    check_array_for_bad(grid, raise_for_bad=raise_for_bad)

    grid_ = grid.copy()

    for i, grid_value in enumerate(grid_):

        if grid_value < reflecting_grid_value:

            grid_[i] += (reflecting_grid_value - grid_value) * 2

        else:

            grid_[i] -= (grid_value - reflecting_grid_value) * 2

    return grid_
