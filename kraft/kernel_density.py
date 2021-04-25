from KDEpy import FFTKDE
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from .CONSTANT import FLOAT_RESOLUTION
from .grid import make_1d, make_nd, plot as grid_plot


def get_bandwidth(vector):

    return KDEMultivariate(vector, "c").bw[0]


def get_density(
    point_x_dimension, bandwidth_=None, _1d_grid_=None, plot=True, dimension_name_=None
):

    dimension_x_point = point_x_dimension.T

    if bandwidth_ is None:

        bandwidth_ = tuple(get_bandwidth(vector) for vector in dimension_x_point)

    if _1d_grid_ is None:

        _1d_grid_ = tuple(
            make_1d(vector.min(), vector.max(), 0.1, 8) for vector in dimension_x_point
        )

    nd_grid = make_nd(_1d_grid_)

    # TODO: consider 0ing
    nd_density_vector = (
        FFTKDE(bw=bandwidth_).fit(point_x_dimension).evaluate(nd_grid)
    ).clip(min=FLOAT_RESOLUTION)

    if plot:

        grid_plot(
            nd_grid,
            nd_density_vector,
            dimension_name_=dimension_name_,
            number_name="Density",
        )

    return nd_grid, nd_density_vector
