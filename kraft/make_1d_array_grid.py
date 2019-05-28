from numpy import linspace


def make_1d_array_grid(_1d_array, fraction_grid_extension, n_grid):

    _1d_array_min = _1d_array.min()

    _1d_array_max = _1d_array.max()

    _1d_array_range = _1d_array_max - _1d_array_min

    extension = _1d_array_range * fraction_grid_extension

    return linspace(_1d_array_min - extension, _1d_array_max + extension, num=n_grid)
