from numpy import unique


def unmesh(point_x_dimension, values):

    grids = tuple(unique(dimension) for dimension in point_x_dimension.T)

    return grids, values.reshape(tuple(grid.size for grid in grids))
