from numpy import absolute, apply_along_axis, argmax, linspace, asarray

from .plot_mesh_grid import plot_mesh_grid
from .compute_posterior_probability import compute_posterior_probability
from .make_mesh_grid_point_x_dimension import make_mesh_grid_point_x_dimension
from .unmesh import unmesh


def _get_target_mesh_grid_point_x_dimension(nd_array, function):

    return tuple(
        make_mesh_grid_point_x_dimension(
            (
                linspace(
                    0,
                    nd_array.shape[i],
                    num=nd_array.shape[i],
                    endpoint=False,
                    dtype=int,
                )
                for i in range(nd_array.ndim - 1)
            )
        ).T
    ) + (apply_along_axis(function, -1, nd_array).ravel(),)


def infer(observation_x_dimension, target="max", plot=True, dimension_names=None):

    n_dimension = observation_x_dimension.shape[1]

    mesh_grid_point_x_dimensiona, mesh_grid_point_posterior_probability = compute_posterior_probability(
        observation_x_dimension, plot=plot, dimension_names=dimension_names
    )

    dimension_grids, posterior_probability = unmesh(
        mesh_grid_point_x_dimensiona, mesh_grid_point_posterior_probability
    )

    if target == "max":

        target_mesh_grid_point_x_dimension = _get_target_mesh_grid_point_x_dimension(
            posterior_probability, argmax
        )

    else:

        i = absolute(dimension_grids[-1] - target).argmin()

        target_mesh_grid_point_x_dimension = _get_target_mesh_grid_point_x_dimension(
            posterior_probability, lambda _: i
        )

    mesh_grid_point_YYY = posterior_probability[target_mesh_grid_point_x_dimension]

    mesh_grid_point_x_XXX = make_mesh_grid_point_x_dimension(dimension_grids[:-1])

    if plot:

        if n_dimension == 2:

            title = f"P({dimension_names[1]} = {target} | {dimension_names[0]})"

        elif n_dimension == 3:

            title = f"P({dimension_names[2]} = {target} | {dimension_names[0]}, {dimension_names[1]})"

        else:

            title = None

        plot_mesh_grid(
            mesh_grid_point_x_XXX,
            mesh_grid_point_YYY,
            title=title,
            dimension_names=dimension_names,
        )

    return posterior_probability  # , p_tvt__ntvs
