from numpy import diff, unique


def get_1d_grid_resolution(co_):
    return diff(unique(co_)).min()
