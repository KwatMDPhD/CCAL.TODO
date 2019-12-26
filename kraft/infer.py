from numpy import absolute, unique

from .compute_element_x_dimension_posterior_probability import (
    compute_element_x_dimension_posterior_probability,
)
from .plot_mesh_grid import plot_mesh_grid


def infer(
    element_x_dimension, target_dimension_value, plot=True, dimension_names=None,
):

    (
        mesh_grid_point_x_dimension,
        mesh_grid_point_posterior_probability,
    ) = compute_element_x_dimension_posterior_probability(
        element_x_dimension, plot=plot, dimension_names=dimension_names,
    )

    target_dimension_grid = unique(mesh_grid_point_x_dimension[:, -1])

    target_value_index = absolute(
        target_dimension_grid - target_dimension_value
    ).argmin()

    no_target_mesh_grid_point_x_dimension = mesh_grid_point_x_dimension[
        target_value_index :: target_dimension_grid.size, :-1
    ]

    no_target_mesh_grid_point_posterior_probability = mesh_grid_point_posterior_probability[
        target_value_index :: target_dimension_grid.size
    ]

    if plot:

        plot_mesh_grid(
            no_target_mesh_grid_point_x_dimension,
            no_target_mesh_grid_point_posterior_probability,
            dimension_names=dimension_names,
            value_name="P({} = {:.2e} (~{}) | {})".format(
                dimension_names[-1],
                target_dimension_grid[target_value_index],
                target_dimension_value,
                *dimension_names[:-1],
            ),
        )

    return (
        no_target_mesh_grid_point_x_dimension,
        no_target_mesh_grid_point_posterior_probability,
    )
