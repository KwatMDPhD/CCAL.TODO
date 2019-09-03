from numpy import diff, unique


def get_mesh_grid_point_x_dimension_d_dimensions(mesh_grid_point_x_dimension):

    return tuple(
        diff(unique(mesh_grid_point_x_dimension[:, i])[:2])[0]
        for i in range(mesh_grid_point_x_dimension.shape[1])
    )
