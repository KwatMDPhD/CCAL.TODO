from KDEpy import FFTKDE

from .compute_bandwidth import compute_bandwidth
from .make_grid import make_grid
from .mesh import mesh
from .plot_mesh import plot_mesh


def estimate_kernel_density(
    point_x_dimension, bandwidths=None, grids=None, plot=True, names=None,
):

    dimension_x_point = point_x_dimension.T

    if bandwidths is None:

        bandwidths = tuple(compute_bandwidth(vector) for vector in dimension_x_point)

    if grids is None:

        grids = tuple(
            make_grid(vector.min(), vector.max(), 0, 8) for vector in dimension_x_point
        )

    point_x_dimension = mesh(grids)

    kernel_densities = (
        FFTKDE(bw=bandwidths).fit(point_x_dimension).evaluate(point_x_dimension)
    )

    if plot:

        plot_mesh(
            point_x_dimension,
            kernel_densities,
            names=names,
            value_name="Kernel Density",
        )

    return point_x_dimension, kernel_densities
