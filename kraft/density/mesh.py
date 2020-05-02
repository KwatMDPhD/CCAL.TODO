from numpy import asarray, meshgrid


def mesh(grids):

    return asarray(
        tuple(vector.ravel() for vector in meshgrid(*grids, indexing="ij"))
    ).T
