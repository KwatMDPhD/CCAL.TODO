from numpy import diff, product, unique

from .estimate_kernel_density import estimate_kernel_density
from .plot_mesh import plot_mesh


def compute_joint_probability(
    element_x_dimension,
    plot=True,
    names=None,
    **estimate_kernel_density_keyword_arguments,
):

    point_x_dimension, kernel_densities = estimate_kernel_density(
        element_x_dimension,
        plot=plot,
        names=names,
        **estimate_kernel_density_keyword_arguments,
    )

    joint_probabilities = kernel_densities / (
        kernel_densities.sum()
        * product(tuple(diff(unique(vector)).min() for vector in point_x_dimension.T))
    )

    if plot:

        plot_mesh(
            point_x_dimension,
            joint_probabilities,
            names=names,
            value_name="Joint Probability",
        )

    return point_x_dimension, joint_probabilities
