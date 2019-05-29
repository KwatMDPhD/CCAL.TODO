from .estimate_kernel_density import estimate_kernel_density
from .plot_2d_mesh_grid import plot_2d_mesh_grid


def compute_joint_probabilities(observation_x_dimension, plot=True, names=None):

    mesh_grid_point_x_dimension, mesh_grid_point_density = estimate_kernel_density(
        observation_x_dimension
    )

    mesh_grid_point_probability = (
        mesh_grid_point_density / mesh_grid_point_density.sum()
    )

    if plot and observation_x_dimension.shape[1] == 2:

        plot_2d_mesh_grid(
            mesh_grid_point_x_dimension,
            mesh_grid_point_probability,
            title_template="P({}, {})",
            names=names,
        )

    return mesh_grid_point_x_dimension, mesh_grid_point_probability
