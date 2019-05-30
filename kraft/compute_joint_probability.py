from .estimate_kernel_density import estimate_kernel_density
from .plot_mesh_grid import plot_mesh_grid


def compute_joint_probability(
    observation_x_dimension,
    dimension_bandwidths=None,
    dimension_grid_mins=None,
    dimension_grid_maxs=None,
    dimension_fraction_grid_extensions=None,
    dimension_n_grids=None,
    plot=True,
    dimension_names=None,
):

    mesh_grid_point_x_dimension, mesh_grid_point_kernel_density = estimate_kernel_density(
        observation_x_dimension,
        dimension_bandwidths=dimension_bandwidths,
        dimension_grid_mins=dimension_grid_mins,
        dimension_grid_maxs=dimension_grid_maxs,
        dimension_fraction_grid_extensions=dimension_fraction_grid_extensions,
        dimension_n_grids=dimension_n_grids,
        plot=False,
        dimension_names=dimension_names,
    )

    mesh_grid_point_joint_probability = (
        mesh_grid_point_kernel_density / mesh_grid_point_kernel_density.sum()
    )

    if plot:

        plot_mesh_grid(
            mesh_grid_point_x_dimension,
            mesh_grid_point_joint_probability,
            title="Joint Probability",
            dimension_names=dimension_names,
        )

    return mesh_grid_point_x_dimension, mesh_grid_point_joint_probability
