from numpy import absolute, unique

from .compute_element_x_dimension_posterior_probability import (
    compute_element_x_dimension_posterior_probability,
)
from .FRACTION_GRID_EXTENSION_FOR_ESTIMATING_KERNEL_DENSITY import (
    FRACTION_GRID_EXTENSION_FOR_ESTIMATING_KERNEL_DENSITY,
)
from .N_GRID_FOR_ESTIMATING_KERNEL_DENSITY import N_GRID_FOR_ESTIMATING_KERNEL_DENSITY
from .plot_mesh_grid import plot_mesh_grid


def infer(
    element_x_dimension,
    target_dimension_value,
    fraction_grid_extension=FRACTION_GRID_EXTENSION_FOR_ESTIMATING_KERNEL_DENSITY,
    n_grid=N_GRID_FOR_ESTIMATING_KERNEL_DENSITY,
    plot=True,
    dimension_names=None,
):

    n_dimension = element_x_dimension.shape[1]

    mesh_grid_point_x_dimension, mesh_grid_point_posterior_probability = compute_element_x_dimension_posterior_probability(
        element_x_dimension,
        plot=plot,
        dimension_names=dimension_names,
        dimension_fraction_grid_extensions=(fraction_grid_extension,) * n_dimension,
        dimension_n_grids=(n_grid,) * n_dimension,
    )

    target__dimension_grid = unique(mesh_grid_point_x_dimension[:, -1])

    target__value_index = absolute(
        target__dimension_grid - target_dimension_value
    ).argmin()

    no_target__mesh_grid_point_x_dimension = mesh_grid_point_x_dimension[
        target__value_index :: target__dimension_grid.size, :-1
    ]

    no_target__mesh_grid_point_posterior_probability = mesh_grid_point_posterior_probability[
        target__value_index :: target__dimension_grid.size
    ]

    if plot:

        target_dimension_value_ = target__dimension_grid[target__value_index]

        if n_dimension == 2:

            layout_title_text = "P({} = {:.3f} (~{}) | {})".format(
                dimension_names[1],
                target_dimension_value_,
                target_dimension_value,
                dimension_names[0],
            )

        elif n_dimension == 3:

            layout_title_text = "P({} = {:.3f} (~{}) | {}, {})".format(
                dimension_names[2],
                target_dimension_value_,
                target_dimension_value,
                dimension_names[0],
                dimension_names[1],
            )

        else:

            layout_title_text = None

        plot_mesh_grid(
            no_target__mesh_grid_point_x_dimension,
            no_target__mesh_grid_point_posterior_probability,
            layout={"title": {"text": layout_title_text}},
            dimension_names=dimension_names,
        )

    return (
        no_target__mesh_grid_point_x_dimension,
        no_target__mesh_grid_point_posterior_probability,
    )
