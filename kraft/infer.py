from numpy import absolute, unique

from .compute_posterior_probability import compute_posterior_probability
from .plot_mesh_grid import plot_mesh_grid


def infer(
    observation_x_dimension,
    target_dimension_value,
    dimension_fraction_grid_extensions=None,
    dimension_n_grids=None,
    plot=True,
    dimension_names=None,
):

    n_dimension = observation_x_dimension.shape[1]

    mesh_grid_point_x_dimension, mesh_grid_point_posterior_probability = compute_posterior_probability(
        observation_x_dimension,
        dimension_fraction_grid_extensions=dimension_fraction_grid_extensions,
        dimension_n_grids=dimension_n_grids,
        plot=plot,
        dimension_names=dimension_names,
    )

    target_dimension_grid = unique(mesh_grid_point_x_dimension[:, -1])

    target_dimension_value_target_dimension_grid_index = absolute(
        target_dimension_grid - target_dimension_value
    ).argmin()

    mesh_grid_point_x_dimension_ = mesh_grid_point_x_dimension[
        target_dimension_value_target_dimension_grid_index :: target_dimension_grid.size,
        :-1,
    ]

    mesh_grid_point_posterior_probability_ = mesh_grid_point_posterior_probability[
        target_dimension_value_target_dimension_grid_index :: target_dimension_grid.size
    ]

    if plot:

        if n_dimension == 2:

            title = f"P({dimension_names[1]} = {target_dimension_value} | {dimension_names[0]})"

        elif n_dimension == 3:

            title = f"P({dimension_names[2]} = {target_dimension_value} | {dimension_names[0]}, {dimension_names[1]})"

        else:

            title = None

        plot_mesh_grid(
            mesh_grid_point_x_dimension_,
            mesh_grid_point_posterior_probability_,
            title=title,
            dimension_names=dimension_names,
        )

    return mesh_grid_point_x_dimension_, mesh_grid_point_posterior_probability_
