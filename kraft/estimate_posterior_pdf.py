from numpy import apply_along_axis, diff, unique

from .estimate_pdf import estimate_pdf
from .plot_mesh import plot_mesh
from .unmesh import unmesh


def estimate_posterior_pdf(
    point_x_dimension,
    plot=True,
    names=None,
    **estimate_kernel_density_keyword_arguments,
):

    point_x_dimension, joint_probabilities = estimate_pdf(
        point_x_dimension,
        plot=plot,
        names=names,
        **estimate_kernel_density_keyword_arguments,
    )

    target_dimension_resolution = diff(unique(point_x_dimension[:, -1])).min()

    posterior_probabilities = apply_along_axis(
        lambda vector: vector / (vector.sum() * target_dimension_resolution),
        -1,
        unmesh(point_x_dimension, joint_probabilities)[1],
    ).reshape(joint_probabilities.shape)

    if plot:

        plot_mesh(
            point_x_dimension,
            posterior_probabilities,
            names=names,
            value_name="Posterior PDF",
        )

    return point_x_dimension, posterior_probabilities
