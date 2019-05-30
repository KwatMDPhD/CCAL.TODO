from .compute_joint_probability import compute_joint_probability
from numpy import absolute


def infer_assuming_independence(
    observation_x_dimension,
    target_dimension_value,
    dimension_fraction_grid_extensions=None,
    dimension_n_grids=None,
    plot=True,
    dimension_names=None,
):

    n_dimension = observation_x_dimension.shape[1]

    mesh_grid_point_x_dimension, mesh_grid_point_joint_probability = compute_joint_probability(
        observation_x_dimension[:, -1:],
        dimension_fraction_grid_extensions=dimension_fraction_grid_extensions[-1:],
        dimension_n_grids=dimension_n_grids[-1:],
        plot=plot,
        dimension_names=dimension_names[-1:],
    )

    print(mesh_grid_point_x_dimension.shape, mesh_grid_point_joint_probability.shape)

    target_dimension_value_target_dimension_grid_index = absolute(
        mesh_grid_point_x_dimension - target_dimension_value
    ).argmin()

    print(target_dimension_value_target_dimension_grid_index)

    # p_tv = kd_tv / kd_tv.sum()
    #
    # grid_tv = linspace(variables[-1].min(), variables[-1].max(), num=n_grid)
    #
    # if target == "max":
    #
    #     t_i = p_tv.argmax()
    #
    # else:
    #
    #     t_i = absolute(grid_tv - target).argmin()
    #
    # t = grid_tv[t_i]
    #
    # p_tvt = p_tv[t_i]
    #
    # if names is None:
    #
    #     names = tuple("Variable {}".format(i) for i in range(n_dimension))
    #
    # if plot:
    #
    #     plot_and_save(
    #         {
    #             "layout": {
    #                 "title": {
    #                     "text": "P({} = {} = {}) = {}".format(
    #                         names[-1], target, t, p_tvt
    #                     )
    #                 },
    #                 "xaxis": {"title": names[-1]},
    #                 "yaxis": {"title": "Probability"},
    #             },
    #             "data": [
    #                 {
    #                     "type": "scatter",
    #                     "x": grid_tv,
    #                     "y": p_tv,
    #                     "marker": {"color": "#20d9ba"},
    #                 }
    #             ],
    #         },
    #         None,
    #     )
    #
    # p_tvt__1ntvs = tuple(
    #     infer(
    #         (variables[ntv_i], variables[-1]),
    #         n_grid=n_grid,
    #         target=t,
    #         plot=False,
    #         names=(names[ntv_i], names[-1]),
    #     )[1]
    #     for ntv_i in range(n_ntv)
    # )
    #
    # p_tvt__ntvs = full((n_grid,) * n_ntv, nan)
    #
    # mesh_index_grid_raveled = make_mesh_grid_and_ravel(
    #     (0,) * n_ntv, (n_grid - 1,) * n_ntv, (n_grid,) * n_ntv
    # )
    #
    # for i in range(n_grid ** n_ntv):
    #
    #     indices = tuple(
    #         [mesh_index_grid_raveled[ntv_i][i].astype(int)] for ntv_i in range(n_ntv)
    #     )
    #
    #     p_tvt__ntvs[indices] = product(
    #         [p_tvt__1ntvs[j][indices[j]] for j in range(n_ntv)]
    #     ) / p_tvt ** (n_ntv - 1)
    #
    # if plot and n_dimension == 3:
    #
    #     plot_heat_map(
    #         DataFrame(rot90(p_tvt__ntvs)),
    #         title="P({} = {} = {} | {}, {})".format(
    #             names[-1], target, t, names[0], names[1]
    #         ),
    #         xaxis_title=names[0],
    #         yaxis_title=names[1],
    #     )
    #
    # return None, p_tvt__ntvs
