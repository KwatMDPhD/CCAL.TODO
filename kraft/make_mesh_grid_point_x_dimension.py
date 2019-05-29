from numpy import asarray, meshgrid


def make_mesh_grid_point_x_dimension(dimension_grids):

    return asarray(
        tuple(
            dimension_mesh_grid.ravel()
            for dimension_mesh_grid in meshgrid(*dimension_grids, indexing="ij")
        )
    ).T
