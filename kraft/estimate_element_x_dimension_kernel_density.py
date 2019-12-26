from KDEpy import FFTKDE

from .compute_vector_bandwidth import compute_vector_bandwidth
from .make_dimension_grid import make_dimension_grid
from .make_mesh_grid_point_x_dimension import make_mesh_grid_point_x_dimension
from .plot_mesh_grid import plot_mesh_grid


def estimate_element_x_dimension_kernel_density(
    element_x_dimension,
    dimension_bandwidths=None,
    dimension_grids=None,
    dimension_grid_mins=None,
    dimension_grid_maxs=None,
    dimension_fraction_grid_extensions=None,
    dimension_n_grids=None,
    plot=True,
    dimension_names=None,
):

    n_dimension = element_x_dimension.shape[1]

    if dimension_bandwidths is None:

        dimension_bandwidths = tuple(
            compute_vector_bandwidth(element_x_dimension[:, i])
            for i in range(n_dimension)
        )

    if dimension_grids is None:

        if dimension_grid_mins is None:

            dimension_grid_mins = tuple(
                element_x_dimension[:, i].min() for i in range(n_dimension)
            )

        if dimension_grid_maxs is None:

            dimension_grid_maxs = tuple(
                element_x_dimension[:, i].max() for i in range(n_dimension)
            )

        if dimension_fraction_grid_extensions is None:

            dimension_fraction_grid_extensions = (0,) * n_dimension

        if dimension_n_grids is None:

            dimension_n_grids = (8,) * n_dimension

        dimension_grids = (
            make_dimension_grid(
                dimension_grid_mins[i],
                dimension_grid_maxs[i],
                dimension_fraction_grid_extensions[i],
                dimension_n_grids[i],
            )
            for i in range(n_dimension)
        )

    mesh_grid_point_x_dimension = make_mesh_grid_point_x_dimension(dimension_grids)

    mesh_grid_point_kernel_density = (
        FFTKDE(bw=dimension_bandwidths)
        .fit(element_x_dimension)
        .evaluate(mesh_grid_point_x_dimension)
    )

    if plot:

        plot_mesh_grid(
            mesh_grid_point_x_dimension,
            mesh_grid_point_kernel_density,
            dimension_names=dimension_names,
            value_name="Kernel Density",
        )

    return mesh_grid_point_x_dimension, mesh_grid_point_kernel_density
