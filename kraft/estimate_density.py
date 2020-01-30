from KDEpy import FFTKDE
from numpy import finfo

from .compute_bandwidth import compute_bandwidth
from .make_grid import make_grid
from .mesh import mesh
from .plot_mesh import plot_mesh


def estimate_density(
    point_x_dimension, bandwidths=None, grids=None, plot=True, names=None,
):

    dimension_x_point = point_x_dimension.T

    if bandwidths is None:

        bandwidths = tuple(compute_bandwidth(vector) for vector in dimension_x_point)

    if grids is None:

        grids = tuple(
            make_grid(vector.min(), vector.max(), 0.1, 8)
            for vector in dimension_x_point
        )

    grid_point_x_dimension = mesh(grids)

    density = (
        FFTKDE(bw=bandwidths).fit(point_x_dimension).evaluate(grid_point_x_dimension)
    ).clip(min=finfo(float).resolution)

    if plot:

        plot_mesh(
            grid_point_x_dimension, density, names=names, value_name="Density",
        )

    return grid_point_x_dimension, density
