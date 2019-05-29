from numpy import apply_along_axis
from .plot_2d_mesh_grid import plot_2d_mesh_grid

from .unmesh import unmesh
from .compute_joint_probability import compute_joint_probability


def compute_posterior_probability(
    observation_x_dimension, n_grids=None, plot=True, dimension_names=None
):

    mesh_grid_point_x_dimension, mesh_grid_point_joint_probability = compute_joint_probability(
        observation_x_dimension,
        n_grids=n_grids,
        plot=plot,
        dimension_names=dimension_names,
    )

    dimension_grids, joint_probability = unmesh(
        mesh_grid_point_x_dimension, mesh_grid_point_joint_probability
    )

    posterior_probability = apply_along_axis(
        lambda _1d_array: _1d_array / _1d_array.sum(), -1, joint_probability
    )

    mesh_grid_point_posterior_probability = posterior_probability.reshape(
        mesh_grid_point_joint_probability.shape
    )

    if plot and observation_x_dimension.shape[1] == 2:

        plot_2d_mesh_grid(
            mesh_grid_point_x_dimension,
            mesh_grid_point_posterior_probability,
            title_template="P({1} | {0})",
            dimension_names=dimension_names,
        )

    return mesh_grid_point_x_dimension, mesh_grid_point_posterior_probability
