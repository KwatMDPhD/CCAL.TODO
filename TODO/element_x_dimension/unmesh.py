from numpy import apply_along_axis, unique


def unmesh(mesh_grid_point_x_dimension, mesh_grid_point_value):

    dimension_grids = apply_along_axis(unique, 0, mesh_grid_point_x_dimension).T

    return (
        dimension_grids,
        mesh_grid_point_value.reshape(
            tuple(dimension_grid.size for dimension_grid in dimension_grids)
        ),
    )
