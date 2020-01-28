from numpy import diff, product, unique

from .estimate_kernel_density import estimate_kernel_density
from .plot_mesh import plot_mesh


def estimate_joint_pdf(
    point_x_dimension,
    plot=True,
    names=None,
    **estimate_kernel_density_keyword_arguments,
):

    point_x_dimension, kernel_density = estimate_kernel_density(
        point_x_dimension,
        plot=plot,
        names=names,
        **estimate_kernel_density_keyword_arguments,
    )

    joint_pdf = kernel_density / (
        kernel_density.sum()
        * product(tuple(diff(unique(vector)).min() for vector in point_x_dimension.T))
    )

    if plot:

        plot_mesh(
            point_x_dimension, joint_pdf, names=names, value_name="Joint PDF",
        )

    return point_x_dimension, joint_pdf
