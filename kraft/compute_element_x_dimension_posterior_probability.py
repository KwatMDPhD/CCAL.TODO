from numpy import apply_along_axis

from .compute_element_x_dimension_joint_probability import (
    compute_element_x_dimension_joint_probability,
)
from .get_element_x_dimension_d_dimensions import get_element_x_dimension_d_dimensions
from .plot_mesh_grid import plot_mesh_grid
from .unmesh import unmesh


def compute_element_x_dimension_posterior_probability(
    element_x_dimension,
    plot=True,
    dimension_names=None,
    **estimate_element_x_dimension_kernel_density_keyword_arguments,
):

    mesh_grid_point_x_dimension, mesh_grid_point_joint_probability = compute_element_x_dimension_joint_probability(
        element_x_dimension,
        plot=plot,
        dimension_names=dimension_names,
        **estimate_element_x_dimension_kernel_density_keyword_arguments,
    )

    d_target_dimension = get_element_x_dimension_d_dimensions(
        mesh_grid_point_x_dimension
    )[-1]

    joint_probability = unmesh(
        mesh_grid_point_x_dimension, mesh_grid_point_joint_probability
    )[1]

    mesh_grid_point_posterior_probability = apply_along_axis(
        lambda vector: vector / (vector.sum() * d_target_dimension),
        -1,
        joint_probability,
    ).reshape(mesh_grid_point_joint_probability.shape)

    if plot:

        plot_mesh_grid(
            mesh_grid_point_x_dimension,
            mesh_grid_point_posterior_probability,
            title_text="Posterior Probability",
            dimension_names=dimension_names,
        )

    return mesh_grid_point_x_dimension, mesh_grid_point_posterior_probability
