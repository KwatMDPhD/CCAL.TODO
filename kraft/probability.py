from numpy import apply_along_axis, product

from .grid import get_1d, get_d, plot as grid_plot
from .kernel_density import get_density


def get_probability(
    point_x_dimension, plot=True, dimension_name_=None, **kwarg_,
):

    nd_grid, nd_density_vector = get_density(
        point_x_dimension, plot=plot, dimension_name_=dimension_name_, **kwarg_,
    )

    nd_probability_vector = nd_density_vector / (
        nd_density_vector.sum() * product(tuple(get_d(vector) for vector in nd_grid.T))
    )

    if plot:

        grid_plot(
            nd_grid,
            nd_probability_vector,
            dimension_name_=dimension_name_,
            number_name="Probability",
        )

    return nd_grid, nd_probability_vector


def _get_probability(vector):

    return vector / vector.sum()


def get_posterior_probability(
    point_x_dimension, plot=True, dimension_name_=None, **kwarg_
):

    nd_grid, nd_probability_vector = get_probability(
        point_x_dimension, plot=plot, dimension_name_=dimension_name_, **kwarg_,
    )

    target_dimension_d = get_d(nd_grid[:, -1])

    nd_probability_array = nd_probability_vector.reshape(
        tuple(_1d_grid.size for _1d_grid in get_1d(nd_grid))
    )

    nd_posterior_probability_array = (
        apply_along_axis(_get_probability, -1, nd_probability_array)
        * target_dimension_d
    )

    nd_posterior_probability_vector = nd_posterior_probability_array.reshape(
        nd_grid.shape
    )

    if dimension_name_ is None:

        dimension_name_ = tuple(
            "Dimension {}".format(index) for index in range(nd_grid.shape[1])
        )

    if plot:

        grid_plot(
            nd_grid,
            nd_posterior_probability_vector,
            dimension_name_=dimension_name_,
            number_name="Posterior Probability",
        )

    return nd_grid, nd_posterior_probability_vector
