from numpy import absolute, apply_along_axis, product

from .compute_element_x_dimension_joint_probability import (
    compute_element_x_dimension_joint_probability,
)
from .infer import infer
from .mesh_grid_point_x_dimension import mesh_grid_point_x_dimension
from .plot_mesh_grid import plot_mesh_grid
from .unmesh import unmesh


def infer_assuming_independence(
    element_x_dimension, target_dimension_value, plot=True, dimension_names=None,
):

    n_dimension = element_x_dimension.shape[1]

    if dimension_names is None:

        dimension_names = tuple(
            "Dimension {} Variable".format(i) for i in range(n_dimension)
        )

    (
        target_mesh_grid_point_x_dimension,
        target_mesh_grid_point_posterior_probability,
    ) = compute_element_x_dimension_joint_probability(
        element_x_dimension[:, -1:], plot=plot, dimension_names=dimension_names[-1:],
    )

    target_dimensino_grids, target_probability = unmesh(
        target_mesh_grid_point_x_dimension, target_mesh_grid_point_posterior_probability
    )

    target_dimension_grid = target_dimensino_grids[0]

    target_value_index = absolute(
        target_dimension_grid - target_dimension_value
    ).argmin()

    infer_returns = tuple(
        infer(
            element_x_dimension[:, [i, -1]],
            target_dimension_value,
            plot=plot,
            dimension_names=(dimension_names[i], dimension_names[-1]),
        )
        for i in range(n_dimension - 1)
    )

    no_target_mesh_grid_point_x_dimension = mesh_grid_point_x_dimension(
        tuple(infer_return[0] for infer_return in infer_returns)
    )

    no_target_mesh_grid_point_posterior_probability = apply_along_axis(
        product,
        1,
        mesh_grid_point_x_dimension(
            tuple(infer_return[1] for infer_return in infer_returns)
        ),
    ) / (target_probability[target_value_index] ** (n_dimension - 2))

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
