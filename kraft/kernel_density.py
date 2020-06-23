from KDEpy import FFTKDE
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from .CONSTANT import FLOAT_RESOLUTION
from .point_x_dimension import make_grid_1d, make_grid_nd, plot_grid_nd


def get_bandwidth(vector):

    return KDEMultivariate(vector, "c").bw[0]


def get_density(point_x_dimension, bandwidths=None, grids=None, plot=True, names=None):

    dimension_x_point = point_x_dimension.T

    if bandwidths is None:

        bandwidths = tuple(get_bandwidth(dimension) for dimension in dimension_x_point)

    if grids is None:

        grids = tuple(
            make_grid_1d(dimension.min(), dimension.max(), 0.1, 8)
            for dimension in dimension_x_point
        )

    grid_point_x_dimension = make_grid_nd(grids)

    grid_point_x_dimension_density = (
        FFTKDE(bw=bandwidths).fit(point_x_dimension).evaluate(grid_point_x_dimension)
    ).clip(min=FLOAT_RESOLUTION)

    if plot:

        plot_grid_nd(
            grid_point_x_dimension,
            grid_point_x_dimension_density,
            dimension_names=names,
            value_name="Density",
        )

    return grid_point_x_dimension, grid_point_x_dimension_density
