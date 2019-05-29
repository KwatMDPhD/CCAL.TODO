from numpy import apply_along_axis

from .compute_joint_probability import compute_joint_probability
from .plot_mesh_grid import plot_mesh_grid
from .unmesh import unmesh


def compute_posterior_probability(
    observation_x_dimension,
    dimension_bandwidths=None,
    dimension_grid_mins=None,
    dimension_grid_maxs=None,
    dimension_fraction_grid_extensions=None,
    dimension_n_grids=None,
    plot=True,
    dimension_names=None,
):

    mesh_grid_point_x_dimension, mesh_grid_point_joint_probability = compute_joint_probability(
        observation_x_dimension,
        dimension_bandwidths=dimension_bandwidths,
        dimension_grid_mins=dimension_grid_mins,
        dimension_grid_maxs=dimension_grid_maxs,
        dimension_fraction_grid_extensions=dimension_fraction_grid_extensions,
        dimension_n_grids=dimension_n_grids,
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

    if plot:

        plot_mesh_grid(
            mesh_grid_point_x_dimension,
            mesh_grid_point_posterior_probability,
            title="Posterior Probability",
            dimension_names=dimension_names,
        )

    return mesh_grid_point_x_dimension, mesh_grid_point_posterior_probability
