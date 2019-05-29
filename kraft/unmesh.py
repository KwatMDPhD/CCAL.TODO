from numpy import asarray, unique


def unmesh(mesh_grid_point_x_dimension, mesh_grid_point_value):

    dimension_grids = asarray(
        tuple(
            unique(mesh_grid_point_x_dimension[:, i])
            for i in range(mesh_grid_point_x_dimension.shape[1])
        )
    )

    return (
        dimension_grids,
        mesh_grid_point_value.reshape(
            tuple(dimension_grid.size for dimension_grid in dimension_grids)
        ),
    )
