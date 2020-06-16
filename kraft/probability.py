from numpy import absolute, apply_along_axis, diff, product, unique

from .kernel_density import get_density
from .point_x_dimension import get_grids, plot_grid_point_x_dimension


def get_pdf(
    point_x_dimension, plot=True, names=None, **estimate_density_keyword_arguments,
):

    grid_point_x_dimension, point_density = get_density(
        point_x_dimension, plot=plot, names=names, **estimate_density_keyword_arguments,
    )

    point_pdf = point_density / (
        point_density.sum()
        * product(
            tuple(
                diff(unique(dimension)).min() for dimension in grid_point_x_dimension.T
            )
        )
    )

    if plot:

        plot_grid_point_x_dimension(
            grid_point_x_dimension, point_pdf, names=names, value_name="PDF",
        )

    return grid_point_x_dimension, point_pdf


def get_posterior_pdf(
    point_x_dimension, plot=True, names=None, **estimate_density_keyword_arguments,
):

    grid_point_x_dimension, point_joint_pdf = get_pdf(
        point_x_dimension, plot=plot, names=names, **estimate_density_keyword_arguments,
    )

    target_dimension_resolution = diff(unique(grid_point_x_dimension[:, -1])).min()

    def get_posterior_probability(array):

        return array / (array.sum() * target_dimension_resolution)

    point_posterior_pdf = apply_along_axis(
        get_posterior_probability,
        -1,
        point_joint_pdf.reshape(
            tuple(grid.size for grid in get_grids(grid_point_x_dimension))
        ),
    ).reshape(point_joint_pdf.shape)

    if plot:

        plot_grid_point_x_dimension(
            grid_point_x_dimension,
            point_posterior_pdf,
            names=names,
            value_name="Posterior PDF",
        )

    return grid_point_x_dimension, point_posterior_pdf


def target_posterior_pdf(
    grid_point_x_dimension, point_posterior_probability, value, plot=True, names=None,
):

    target_dimension_grid = unique(grid_point_x_dimension[:, -1])

    target_value_index = absolute(target_dimension_grid - value).argmin()

    grid_point_x_dimension_ = grid_point_x_dimension[
        target_value_index :: target_dimension_grid.size, :-1
    ]

    point_posterior_probability_ = point_posterior_probability[
        target_value_index :: target_dimension_grid.size
    ]

    if plot:

        plot_grid_point_x_dimension(
            grid_point_x_dimension_,
            point_posterior_probability_,
            names=names,
            value_name="P({} = {:.2e} (~{}) | {})".format(
                names[-1],
                target_dimension_grid[target_value_index],
                value,
                *names[:-1],
            ),
        )

    return grid_point_x_dimension_, point_posterior_probability_
