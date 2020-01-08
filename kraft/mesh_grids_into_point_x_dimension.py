from numpy import asarray, meshgrid


def mesh_grids_into_point_x_dimension(grids):

    return asarray(
        tuple(
            dimension_values.ravel()
            for dimension_values in meshgrid(*grids, indexing="ij")
        )
    ).T
