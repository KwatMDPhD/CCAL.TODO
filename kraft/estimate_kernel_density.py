from KDEpy import FFTKDE
from .make_mesh_grid_point_x_dimension import make_mesh_grid_point_x_dimension
from .compute_1d_array_bandwidth import compute_1d_array_bandwidth
from .ALMOST_ZERO import ALMOST_ZERO
from .make_1d_array_grid import make_1d_array_grid


def estimate_kernel_density(
    observation_x_dimension, bandwidths=None, fraction_grid_extension=1 / 3, n_grid=64
):

    if bandwidths is None:

        bandwidths = tuple(
            compute_1d_array_bandwidth(observation_x_dimension[:, i])
            for i in range(observation_x_dimension.shape[1])
        )

    mesh_grid_point_x_dimension = make_mesh_grid_point_x_dimension(
        tuple(
            make_1d_array_grid(
                observation_x_dimension[:, i], fraction_grid_extension, n_grid
            )
            for i in range(observation_x_dimension.shape[1])
        )
    )

    mesh_grid_point_density = (
        (
            FFTKDE(bw=bandwidths)
            .fit(observation_x_dimension)
            .evaluate(mesh_grid_point_x_dimension)
        )
    ).clip(min=ALMOST_ZERO)

    return mesh_grid_point_x_dimension, mesh_grid_point_density
