from KDEpy import FFTKDE
from numpy import asarray, finfo

from .get_bandwidth import get_bandwidth
from .make_grid import make_grid
from .mesh import mesh
from .plot_mesh import plot_mesh


def estimate_density(
    point_x_dimension,
    bandwidths=None,
    bandwidth_factors=None,
    grids=None,
    plot=True,
    names=None,
):

    dimension_x_point = point_x_dimension.T

    if bandwidths is None:

        bandwidths = tuple(get_bandwidth(dimension) for dimension in dimension_x_point)

    if bandwidth_factors is not None:

        bandwidths = tuple(asarray(bandwidths) * asarray(bandwidth_factors))

    if grids is None:

        grids = tuple(
            make_grid(dimension.min(), dimension.max(), 0.1, 8)
            for dimension in dimension_x_point
        )

    grid_point_x_dimension = mesh(grids)

    density = (
        FFTKDE(bw=bandwidths).fit(point_x_dimension).evaluate(grid_point_x_dimension)
    ).clip(min=finfo(float).resolution)

    if plot:

        plot_mesh(grid_point_x_dimension, density, names=names, value_name="Density")

    return grid_point_x_dimension, density
