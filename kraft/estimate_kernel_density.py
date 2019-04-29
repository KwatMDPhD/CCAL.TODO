from statsmodels.nonparametric.kernel_density import KDEMultivariate

from .make_mesh_grid_and_ravel import make_mesh_grid_and_ravel


def estimate_kernel_density(
    variables, bandwidths=None, mins=None, maxs=None, n_grid=64
):

    n_dimension = len(variables)

    kde_multivariate = KDEMultivariate(variables, "c" * n_dimension, bw=bandwidths)

    if mins is None:

        mins = tuple(variable.min() for variable in variables)

    if maxs is None:

        maxs = tuple(variable.max() for variable in variables)

    n_grids = (n_grid,) * n_dimension

    return kde_multivariate.pdf(make_mesh_grid_and_ravel(mins, maxs, n_grids)).reshape(
        n_grids
    )
