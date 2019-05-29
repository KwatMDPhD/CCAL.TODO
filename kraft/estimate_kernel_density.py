from KDEpy import FFTKDE

from .ALMOST_ZERO import ALMOST_ZERO
from .compute_1d_array_bandwidth import compute_1d_array_bandwidth
from .FRACTION_GRID_EXTENSION import FRACTION_GRID_EXTENSION
from .make_1d_array_grid import make_1d_array_grid
from .make_mesh_grid_point_x_dimension import make_mesh_grid_point_x_dimension
from .N_GRID import N_GRID
from .plot_2d_mesh_grid import plot_2d_mesh_grid


def estimate_kernel_density(
    observation_x_dimension,
    bandwidths=None,
    grid_mins=None,
    grid_maxs=None,
    fraction_grid_extensions=None,
    n_grids=None,
    plot=True,
    names=None,
):

    n_dimension = observation_x_dimension.shape[1]

    if bandwidths is None:

        bandwidths = tuple(
            compute_1d_array_bandwidth(observation_x_dimension[:, i])
            for i in range(n_dimension)
        )

    if grid_mins is None:

        grid_mins = (None,) * n_dimension

    if grid_maxs is None:

        grid_maxs = (None,) * n_dimension

    if fraction_grid_extensions is None:

        fraction_grid_extensions = (FRACTION_GRID_EXTENSION,) * n_dimension

    if n_grids is None:

        n_grids = (N_GRID,) * n_dimension

    mesh_grid_point_x_dimension = make_mesh_grid_point_x_dimension(
        (
            make_1d_array_grid(
                observation_x_dimension[:, i],
                grid_mins[i],
                grid_maxs[i],
                fraction_grid_extensions[i],
                n_grids[i],
            )
            for i in range(n_dimension)
        )
    )

    mesh_grid_point_density = (
        (
            FFTKDE(bw=bandwidths)
            .fit(observation_x_dimension)
            .evaluate(mesh_grid_point_x_dimension)
        )
    ).clip(min=ALMOST_ZERO)

    if plot and observation_x_dimension.shape[1] == 2:

        plot_2d_mesh_grid(
            mesh_grid_point_x_dimension,
            mesh_grid_point_density,
            title_template="D({}, {})",
            names=names,
        )

    return mesh_grid_point_x_dimension, mesh_grid_point_density
