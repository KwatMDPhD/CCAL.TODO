from numpy import asarray, meshgrid


def mesh(grids):

    return asarray(
        tuple(
            dimension_values.ravel()
            for dimension_values in meshgrid(*grids, indexing="ij")
        )
    ).T
