from numpy import absolute, apply_along_axis, product

from .compute_joint_probability import compute_joint_probability
from .infer import infer
from .make_mesh_grid_point_x_dimension import make_mesh_grid_point_x_dimension
from .plot_mesh_grid import plot_mesh_grid


def infer_assuming_independence(
    observation_x_dimension,
    target_dimension_value,
    dimension_fraction_grid_extensions=None,
    dimension_n_grids=None,
    plot=True,
    dimension_names=None,
):

    n_dimension = observation_x_dimension.shape[1]

    target_dimension_grid, target_dimension_probability = compute_joint_probability(
        observation_x_dimension[:, -1:],
        dimension_fraction_grid_extensions=dimension_fraction_grid_extensions[-1:],
        dimension_n_grids=dimension_n_grids[-1:],
        plot=plot,
        dimension_names=dimension_names[-1:],
    )

    infer_returns = tuple(
        infer(
            observation_x_dimension[:, [i, -1]],
            target_dimension_value,
            dimension_fraction_grid_extensions=(
                dimension_fraction_grid_extensions[i],
                dimension_fraction_grid_extensions[-1],
            ),
            dimension_n_grids=(dimension_n_grids[i], dimension_n_grids[-1]),
            plot=False,
            dimension_names=(dimension_names[i], dimension_names[-1]),
        )
        for i in range(n_dimension - 1)
    )

    mesh_grid_point_x_dimension = make_mesh_grid_point_x_dimension(
        tuple(infer_return[0] for infer_return in infer_returns)
    )

    mesh_grid_point_posterior_probability = apply_along_axis(
        product,
        1,
        make_mesh_grid_point_x_dimension(
            tuple(infer_return[1] for infer_return in infer_returns)
        ),
    ) / (
        target_dimension_probability[
            absolute(target_dimension_grid - target_dimension_value).argmin()
        ]
        ** (n_dimension - 2)
    )

    if plot:

        if n_dimension == 2:

            title = f"P({dimension_names[1]} = {target_dimension_value} | {dimension_names[0]})"

        elif n_dimension == 3:

            title = f"P({dimension_names[2]} = {target_dimension_value} | {dimension_names[0]}, {dimension_names[1]})"

        else:

            title = None

        plot_mesh_grid(
            mesh_grid_point_x_dimension,
            mesh_grid_point_posterior_probability,
            title=title,
            dimension_names=dimension_names,
        )

    return mesh_grid_point_x_dimension, mesh_grid_point_posterior_probability
