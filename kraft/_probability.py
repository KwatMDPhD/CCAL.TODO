from numpy import absolute, apply_along_axis, diff, product, unique

from ..kernel_density.estimate_density import estimate_density
from ..point_x_dimension.plot_mesh import plot_mesh
from ..point_x_dimension.unmesh import get_grid
from .estimate_pdf import estimate_pdf


def estimate_pdf(
    point_x_dimension, plot=True, names=None, **estimate_density_keyword_arguments,
):

    grid_point_x_dimension, kernel_density = estimate_density(
        point_x_dimension, plot=plot, names=names, **estimate_density_keyword_arguments,
    )

    pdf = kernel_density / (
        kernel_density.sum()
        * product(
            tuple(
                diff(unique(dimension)).min() for dimension in grid_point_x_dimension.T
            )
        )
    )

    if plot:

        plot_mesh(
            grid_point_x_dimension, pdf, names=names, value_name="PDF",
        )

    return grid_point_x_dimension, pdf


def estimate_posterior_pdf(
    point_x_dimension, plot=True, names=None, **estimate_density_keyword_arguments,
):

    grid_point_x_dimension, joint_pdf = estimate_pdf(
        point_x_dimension, plot=plot, names=names, **estimate_density_keyword_arguments,
    )

    target_dimension_resolution = diff(unique(grid_point_x_dimension[:, -1])).min()

    def get_posterior_probability(array):

        return array / (array.sum() * target_dimension_resolution)

    posterior_pdf = apply_along_axis(
        get_posterior_probability, -1, get_grid(grid_point_x_dimension, joint_pdf)[1]
    ).reshape(joint_pdf.shape)

    if plot:

        plot_mesh(
            grid_point_x_dimension,
            posterior_pdf,
            names=names,
            value_name="Posterior PDF",
        )

    return grid_point_x_dimension, posterior_pdf


def target_posterior_pdf(
    mesh_grid_point_x_dimension,
    mesh_grid_point_posterior_probability,
    value,
    plot=True,
    names=None,
):

    target_dimension_grid = unique(mesh_grid_point_x_dimension[:, -1])

    target_value_index = absolute(target_dimension_grid - value).argmin()

    mesh_grid_point_x_dimension_ = mesh_grid_point_x_dimension[
        target_value_index :: target_dimension_grid.size, :-1
    ]

    mesh_grid_point_posterior_probability_ = mesh_grid_point_posterior_probability[
        target_value_index :: target_dimension_grid.size
    ]

    if plot:

        plot_mesh(
            mesh_grid_point_x_dimension_,
            mesh_grid_point_posterior_probability_,
            names=names,
            value_name="P({} = {:.2e} (~{}) | {})".format(
                names[-1],
                target_dimension_grid[target_value_index],
                value,
                *names[:-1],
            ),
        )

    return mesh_grid_point_x_dimension_, mesh_grid_point_posterior_probability_
