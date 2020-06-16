from numpy import apply_along_axis, diff, unique

from ..kernel_density.plot_mesh import plot_mesh
from ..kernel_density.unmesh import unmesh
from .estimate_pdf import estimate_pdf


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
        get_posterior_probability, -1, unmesh(grid_point_x_dimension, joint_pdf)[1]
    ).reshape(joint_pdf.shape)

    if plot:

        plot_mesh(
            grid_point_x_dimension,
            posterior_pdf,
            names=names,
            value_name="Posterior PDF",
        )

    return grid_point_x_dimension, posterior_pdf
