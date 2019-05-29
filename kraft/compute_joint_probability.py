from .estimate_density import estimate_density
from .plot_2d_mesh_grid import plot_2d_mesh_grid


def compute_joint_probability(
    observation_x_dimension, n_grids=None, plot=True, dimension_names=None
):

    mesh_grid_point_x_dimension, mesh_grid_point_density = estimate_density(
        observation_x_dimension,
        n_grids=n_grids,
        plot=plot,
        dimension_names=dimension_names,
    )

    mesh_grid_point_joint_probability = (
        mesh_grid_point_density / mesh_grid_point_density.sum()
    )

    if plot and observation_x_dimension.shape[1] == 2:

        plot_2d_mesh_grid(
            mesh_grid_point_x_dimension,
            mesh_grid_point_joint_probability,
            title_template="P({}, {})",
            dimension_names=dimension_names,
        )

    return mesh_grid_point_x_dimension, mesh_grid_point_joint_probability
