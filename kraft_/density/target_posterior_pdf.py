from numpy import absolute, unique

from .plot_mesh import plot_mesh


def target_posterior_pdf(
    mesh_grid_point_x_dimension,
    mesh_grid_point_posterior_probability,
    value,
    plot=True,
    names=None,
):

    target_dimension_grid = unique(mesh_grid_point_x_dimension[:, -1])

    target_value_index = absolute(target_dimension_grid - value).argmin()

    mesh_grid_point_x_dimension_ = mesh_grid_point_x_dimension[
        target_value_index :: target_dimension_grid.size, :-1
    ]

    mesh_grid_point_posterior_probability_ = mesh_grid_point_posterior_probability[
        target_value_index :: target_dimension_grid.size
    ]

    if plot:

        plot_mesh(
            mesh_grid_point_x_dimension_,
            mesh_grid_point_posterior_probability_,
            names=names,
            value_name="P({} = {:.2e} (~{}) | {})".format(
                names[-1],
                target_dimension_grid[target_value_index],
                value,
                *names[:-1],
            ),
        )

    return mesh_grid_point_x_dimension_, mesh_grid_point_posterior_probability_
