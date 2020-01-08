from KDEpy import FFTKDE

from .compute_vector_bandwidth import compute_vector_bandwidth
from .make_dimension_grid import make_grid
from .mesh_dimension_grids_into_mesh_grid_point_x_dimension import (
    mesh_dimension_grids_into_mesh_grid_point_x_dimension,
)
from .plot_mesh_grid import plot_mesh_grid


def estimate_element_x_dimension_kernel_density(
    element_x_dimension,
    dimension_bandwidths=None,
    dimension_grids=None,
    plot=True,
    dimension_names=None,
):

    if dimension_bandwidths is None:

        dimension_bandwidths = tuple(
            compute_vector_bandwidth(element_x_dimension[:, i])
            for i in range(element_x_dimension.shape[1])
        )

    if dimension_grids is None:

        dimension_grids = tuple(
            make_grid(
                element_x_dimension[:, i].min(), element_x_dimension[:, i].max(), 0, 8,
            )
            for i in range(element_x_dimension.shape[1])
        )

    mesh_grid_point_x_dimension = mesh_dimension_grids_into_mesh_grid_point_x_dimension(
        dimension_grids
    )

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
