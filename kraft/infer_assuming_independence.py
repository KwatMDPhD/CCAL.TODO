from numpy import absolute, full, linspace, nan, product
from pandas import DataFrame

from .estimate_kernel_density import estimate_kernel_density
from .infer import infer
from .make_mesh_grid_point_x_dimension import make_mesh_grid_point_x_dimension
from .N_GRID import N_GRID
from .plot_and_save import plot_and_save
from .plot_heat_map import plot_heat_map
from .unmesh import unmesh


def infer_assuming_independence(
    observation_x_dimension, n_grid=N_GRID, target="max", plot=True, names=None
):

    n_dimension = observation_x_dimension.shape[1]

    n_ntv = n_dimension - 1

    kd_tv = unmesh(
        estimate_kernel_density(observation_x_dimension[:, -1].reshape(-1, 1)),
        n_grid=n_grid,
    )[1]

    p_tv = kd_tv / kd_tv.sum()

    grid_tv = linspace(
        observation_x_dimension[:, -1].min(),
        observation_x_dimension[:, -1].max(),
        num=n_grid,
    )

    if target == "max":

        t_i = p_tv.argmax()

    else:

        t_i = absolute(grid_tv - target).argmin()

    t = grid_tv[t_i]

    p_tvt = p_tv[t_i]

    if names is None:

        names = tuple(f"Variable {i}" for i in range(n_dimension))

    if plot:

        plot_and_save(
            {
                "layout": {
                    "title": {"text": f"P({names[-1]} = {target} = {t}) = {p_tvt}"},
                    "xaxis": {"title": names[-1]},
                    "yaxis": {"title": "Probability"},
                },
                "data": [
                    {
                        "type": "scatter",
                        "x": grid_tv,
                        "y": p_tv,
                        "marker": {"color": "#20d9ba"},
                    }
                ],
            },
            None,
        )

    p_tvt__1ntvs = tuple(
        infer(
            (variables[ntv_i], variables[-1]),
            n_grid=n_grid,
            target=t,
            plot=False,
            names=(names[ntv_i], names[-1]),
        )[1]
        for ntv_i in range(n_ntv)
    )

    p_tvt__ntvs = full((n_grid,) * n_ntv, nan)

    mesh_grid_point_x_dimension = make_mesh_grid_point_x_dimension(
        linspace(0, n_grid, num=n_grid, endpoint=False, dtype=int) for i in range(n_ntv)
    )

    for i in range(n_grid ** n_ntv):

        indices = tuple(
            [mesh_grid_point_x_dimension[i][ntv_i]] for ntv_i in range(n_ntv)
        )

        p_tvt__ntvs[indices] = product(
            [p_tvt__1ntvs[j][indices[j]] for j in range(n_ntv)]
        ) / p_tvt ** (n_ntv - 1)

    if plot and n_dimension == 3:

        plot_heat_map(
            DataFrame(p_tvt__ntvs),
            title=f"P({names[2]} = {target} = {t} | {names[0]}, {names[1]})",
            xaxis_title=names[1],
            yaxis_title=names[0],
        )

    return None, p_tvt__ntvs
