from numpy import asarray, meshgrid


def make_mesh_grid_point_x_dimension(axis_grids, indexing="ij", raise_for_bad=True):

    return asarray(
        tuple(
            axis_mesh_grid.ravel()
            for axis_mesh_grid in meshgrid(*axis_grids, indexing=indexing)
        )
    ).T
