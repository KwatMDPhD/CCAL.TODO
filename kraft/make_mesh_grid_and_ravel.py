from numpy import linspace, meshgrid


def make_mesh_grid_and_ravel(mins, maxs, n_grids, indexing="ij", raise_for_bad=True):

    return tuple(
        mesh_grid.ravel()
        for mesh_grid in meshgrid(
            *(
                linspace(min_, max_, num=n_grid)
                for min_, max_, n_grid in zip(mins, maxs, n_grids)
            ),
            indexing=indexing
        )
    )
