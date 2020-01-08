from numpy import product

from .estimate_kernel_density import estimate_kernel_density
from .get_resolutions import get_resolutions
from .plot_mesh_grid import plot_mesh_grid


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
        kernel_densities.sum() * product(get_resolutions(point_x_dimension))
    )

    if plot:

        plot_mesh_grid(
            point_x_dimension,
            joint_probabilities,
            names=names,
            value_name="Joint Probability",
        )

    return point_x_dimension, joint_probabilities
