from numpy import linspace, meshgrid, asarray


def make_mesh_grid_point_x_dimension(
    mins, maxs, n_grids, indexing="ij", raise_for_bad=True
):

    return asarray(
        tuple(
            mesh_grid.ravel()
            for mesh_grid in meshgrid(
                *(
                    linspace(min_, max_, num=n_grid)
                    for min_, max_, n_grid in zip(mins, maxs, n_grids)
                ),
                indexing=indexing
            )
        )
    ).T
