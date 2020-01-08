from numpy import product

from .estimate_kernel_density import estimate_kernel_density
from .get_dimension_resolutions import get_dimension_resolutions
from .plot_mesh_grid import plot_mesh_grid


def compute_joint_probability(
    element_x_dimension,
    plot=True,
    dimension_names=None,
    **estimate_kernel_density_keyword_arguments,
):

    point_x_dimension, point_kernel_density = estimate_kernel_density(
        element_x_dimension,
        plot=plot,
        dimension_names=dimension_names,
        **estimate_kernel_density_keyword_arguments,
    )

    point_joint_probability = point_kernel_density / (
        point_kernel_density.sum()
        * product(get_dimension_resolutions(point_x_dimension))
    )

    if plot:

        plot_mesh_grid(
            point_x_dimension,
            point_joint_probability,
            dimension_names=dimension_names,
            value_name="Joint Probability",
        )

    return point_x_dimension, point_joint_probability
