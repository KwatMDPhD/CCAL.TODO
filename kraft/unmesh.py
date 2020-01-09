from numpy import unique


def unmesh(point_x_dimension, values):

    grids = tuple(unique(vector) for vector in point_x_dimension.T)

    return (
        grids,
        values.reshape(tuple(grid.size for grid in grids)),
    )
