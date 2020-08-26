from KDEpy import FFTKDE
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from .CONSTANT import FLOAT_RESOLUTION
from .grid import make_grid_1d, make_grid_nd, plot_grid_nd


def get_bandwidth(vector):

    return KDEMultivariate(vector, "c").bw[0]


def get_density(
    point_x_dimension, bandwidths=None, grid_1ds=None, plot=True, dimension_names=None
):

    dimension_x_point = point_x_dimension.T

    if bandwidths is None:

        bandwidths = tuple(get_bandwidth(dimension) for dimension in dimension_x_point)

    if grid_1ds is None:

        grid_1ds = tuple(
            make_grid_1d(dimension.min(), dimension.max(), 0.1, 8)
            for dimension in dimension_x_point
        )

    grid_nd = make_grid_nd(grid_1ds)

    grid_nd_densities = (
        FFTKDE(bw=bandwidths).fit(point_x_dimension).evaluate(grid_nd)
    ).clip(min=FLOAT_RESOLUTION)

    if plot:

        plot_grid_nd(
            grid_nd,
            grid_nd_densities,
            dimension_names=dimension_names,
            number_name="Density",
        )

    return grid_nd, grid_nd_densities
