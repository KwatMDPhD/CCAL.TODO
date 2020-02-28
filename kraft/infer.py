from numpy import absolute, unique

from .estimate_posterior_pdf import estimate_posterior_pdf
from .plot_mesh import plot_mesh


def infer(
    point_x_dimension,
    value,
    plot=True,
    names=None,
    **estimate_density_keyword_arguments,
):

    (
        mesh_grid_point_x_dimension,
        mesh_grid_point_posterior_probability,
    ) = estimate_posterior_pdf(
        point_x_dimension, plot=plot, names=names, **estimate_density_keyword_arguments
    )

    target_dimension_grid = unique(mesh_grid_point_x_dimension[:, -1])

    target_value_index = absolute(target_dimension_grid - value).argmin()

    no_target_mesh_grid_point_x_dimension = mesh_grid_point_x_dimension[
        target_value_index :: target_dimension_grid.size, :-1
    ]

    no_target_mesh_grid_point_posterior_probability = mesh_grid_point_posterior_probability[
        target_value_index :: target_dimension_grid.size
    ]

    if plot:

        plot_mesh(
            no_target_mesh_grid_point_x_dimension,
            no_target_mesh_grid_point_posterior_probability,
            names=names,
            value_name="P({} = {:.2e} (~{}) | {})".format(
                names[-1],
                target_dimension_grid[target_value_index],
                value,
                *names[:-1],
            ),
        )

    return (
        no_target_mesh_grid_point_x_dimension,
        no_target_mesh_grid_point_posterior_probability,
    )
