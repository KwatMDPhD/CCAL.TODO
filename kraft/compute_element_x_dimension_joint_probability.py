from numpy import product

from .estimate_element_x_dimension_kernel_density import (
    estimate_element_x_dimension_kernel_density,
)
from .get_point_x_dimension_d_dimensions import get_point_x_dimension_d_dimensions
from .plot_mesh_grid import plot_mesh_grid


def compute_element_x_dimension_joint_probability(
    element_x_dimension,
    plot=True,
    dimension_names=None,
    **estimate_element_x_dimension_kernel_density_keyword_arguments,
):

    (
        point_x_dimension,
        point_kernel_density,
    ) = estimate_element_x_dimension_kernel_density(
        element_x_dimension,
        plot=plot,
        dimension_names=dimension_names,
        **estimate_element_x_dimension_kernel_density_keyword_arguments,
    )

    d_dimensions = get_point_x_dimension_d_dimensions(point_x_dimension)

    point_joint_probability = point_kernel_density / (
        point_kernel_density.sum() * product(d_dimensions)
    )

    if plot:

        plot_mesh_grid(
            point_x_dimension,
            point_joint_probability,
            dimension_names=dimension_names,
            value_name="Joint Probability",
        )

    return point_x_dimension, point_joint_probability
