from numpy import absolute, apply_along_axis, arange, argmax, linspace
from pandas import DataFrame
from .unmesh import unmesh

from .compute_posterior_probability import compute_posterior_probability
from .plot_and_save import plot_and_save
from .plot_heat_map import plot_heat_map
from .make_mesh_grid_point_x_dimension import make_mesh_grid_point_x_dimension


def _get_target_index_grid(nd_array, function):

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


def infer(
    observation_x_dimension, n_grids=None, target="max", plot=True, dimension_names=None
):

    n_dimension = observation_x_dimension.shape[1]

    dimension_grids, p_tv__ntvs = unmesh(
        compute_posterior_probability(
            observation_x_dimension,
            n_grids=n_grids,
            plot=plot,
            dimension_names=dimension_names,
        )
    )

    if target == "max":

        t_index_grid = _get_target_index_grid(p_tv__ntvs, argmax)

    else:

        t_grid = dimension_grids[-1]

        t_i = absolute(t_grid - target).argmin()

        t_index_grid = _get_target_index_grid(p_tv__ntvs, lambda _: t_i)

    p_tvt__ntvs = p_tv__ntvs[t_index_grid].reshape(n_grids[:-1])

    if plot:

        if names is None:

            names = tuple(f"Variable {i}" for i in range(n_dimension))

        if n_dimension == 2:

            plot_and_save(
                {
                    "layout": {
                        "title": {"text": f"P({names[-1]} = {target} | {names[0]})"},
                        "xaxis": {"title": names[0]},
                        "yaxis": {"title": "Probability"},
                    },
                    "data": [
                        {
                            "type": "scatter",
                            "x": arange(n_grid),
                            "y": p_tvt__ntvs,
                            "marker": {"color": "#20d9ba"},
                        }
                    ],
                },
                None,
            )

        elif n_dimension == 3:

            plot_heat_map(
                DataFrame(p_tvt__ntvs),
                title=f"P({names[2]} = {target} | {names[0]}, {names[1]})",
                xaxis_title=names[1],
                yaxis_title=names[0],
            )

    return p_tv__ntvs, p_tvt__ntvs
