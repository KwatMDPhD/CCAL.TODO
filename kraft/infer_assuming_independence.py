from numpy import absolute, apply_along_axis, product

from .compute_joint_probability import compute_joint_probability
from .FRACTION_GRID_EXTENSION import FRACTION_GRID_EXTENSION
from .infer import infer
from .make_mesh_grid_point_x_dimension import make_mesh_grid_point_x_dimension
from .N_GRID import N_GRID
from .plot_mesh_grid import plot_mesh_grid
from .unmesh import unmesh


def infer_assuming_independence(
    observation_x_dimension,
    target_dimension_value,
    fraction_grid_extension=FRACTION_GRID_EXTENSION,
    n_grid=N_GRID,
    plot=True,
    dimension_names=None,
):

    n_dimension = observation_x_dimension.shape[1]

    target__mesh_grid_point_x_dimension, target__mesh_grid_point_posterior_probability = compute_joint_probability(
        observation_x_dimension[:, -1:],
        plot=plot,
        dimension_names=dimension_names[-1:],
        dimension_fraction_grid_extensions=(fraction_grid_extension,),
        dimension_n_grids=(n_grid,),
    )

    target__dimensino_grids, target__probability = unmesh(
        target__mesh_grid_point_x_dimension,
        target__mesh_grid_point_posterior_probability,
    )
    print(target__probability.shape)

    target__dimension_grid = target__dimensino_grids[0]

    target__value_index = absolute(
        target__dimension_grid - target_dimension_value
    ).argmin()

    infer_returns = tuple(
        infer(
            observation_x_dimension[:, [i, -1]],
            target_dimension_value,
            fraction_grid_extension=fraction_grid_extension,
            n_grid=n_grid,
            plot=plot,
            dimension_names=(dimension_names[i], dimension_names[-1]),
        )
        for i in range(n_dimension - 1)
    )

    no_target__mesh_grid_point_x_dimension = make_mesh_grid_point_x_dimension(
        tuple(infer_return[0] for infer_return in infer_returns)
    )

    no_target__mesh_grid_point_posterior_probability = apply_along_axis(
        product,
        1,
        make_mesh_grid_point_x_dimension(
            tuple(infer_return[1] for infer_return in infer_returns)
        ),
    ) / (target__probability[target__value_index] ** (n_dimension - 2))

    if plot:

        target_dimension_value_ = target__dimension_grid[target__value_index]

        if n_dimension == 2:

            title = f"P({dimension_names[1]} = {target_dimension_value_:.3f} (~{target_dimension_value}) | {dimension_names[0]})"

        elif n_dimension == 3:

            title = f"P({dimension_names[2]} = {target_dimension_value_:.3f} (~{target_dimension_value}) | {dimension_names[0]}, {dimension_names[1]})"

        else:

            title = None

        plot_mesh_grid(
            no_target__mesh_grid_point_x_dimension,
            no_target__mesh_grid_point_posterior_probability,
            title_text=title,
            dimension_names=dimension_names,
        )

    return (
        no_target__mesh_grid_point_x_dimension,
        no_target__mesh_grid_point_posterior_probability,
    )
