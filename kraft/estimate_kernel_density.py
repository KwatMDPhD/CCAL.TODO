from KDEpy import FFTKDE

from .ALMOST_ZERO import ALMOST_ZERO
from .compute_1d_array_bandwidth import compute_1d_array_bandwidth
from .FRACTION_GRID_EXTENSION import FRACTION_GRID_EXTENSION
from .make_1d_array_grid import make_1d_array_grid
from .make_mesh_grid_point_x_dimension import make_mesh_grid_point_x_dimension
from .N_GRID import N_GRID
from .plot_mesh_grid import plot_mesh_grid


def estimate_kernel_density(
    observation_x_dimension,
    dimension_bandwidths=None,
    dimension_bandwidth_factors=None,
    dimension_grid_mins=None,
    dimension_grid_maxs=None,
    dimension_fraction_grid_extensions=None,
    dimension_n_grids=None,
    plot=True,
    dimension_names=None,
):

    n_dimension = observation_x_dimension.shape[1]

    if dimension_bandwidths is None:

        dimension_bandwidths = tuple(
            compute_1d_array_bandwidth(observation_x_dimension[:, i])
            for i in range(n_dimension)
        )

    if dimension_bandwidth_factors is not None:

        dimension_bandwidths = tuple(
            dimension_bandwidths[i] * dimension_bandwidth_factors[i]
            for i in range(n_dimension)
        )

    if dimension_grid_mins is None:

        dimension_grid_mins = (None,) * n_dimension

    if dimension_grid_maxs is None:

        dimension_grid_maxs = (None,) * n_dimension

    if dimension_fraction_grid_extensions is None:

        dimension_fraction_grid_extensions = (FRACTION_GRID_EXTENSION,) * n_dimension

    if dimension_n_grids is None:

        dimension_n_grids = (N_GRID,) * n_dimension

    mesh_grid_point_x_dimension = make_mesh_grid_point_x_dimension(
        (
            make_1d_array_grid(
                observation_x_dimension[:, i],
                dimension_grid_mins[i],
                dimension_grid_maxs[i],
                dimension_fraction_grid_extensions[i],
                dimension_n_grids[i],
            )
            for i in range(n_dimension)
        )
    )

    mesh_grid_point_kernel_density = (
        (
            FFTKDE(bw=dimension_bandwidths)
            .fit(observation_x_dimension)
            .evaluate(mesh_grid_point_x_dimension)
        )
    ).clip(min=ALMOST_ZERO)

    if plot:

        plot_mesh_grid(
            mesh_grid_point_x_dimension,
            mesh_grid_point_kernel_density,
            title_text="Kernel Density",
            dimension_names=dimension_names,
        )

    return mesh_grid_point_x_dimension, mesh_grid_point_kernel_density
