from numpy import unique


def unmesh(point_x_dimension, point_value):

    grids = tuple(unique(dimension_values) for dimension_values in point_x_dimension.T)

    return (
        grids,
        point_value.reshape(tuple(grid.size for grid in grids)),
    )
