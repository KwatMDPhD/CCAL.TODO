from numpy import absolute, apply_along_axis, product

from .estimate_pdf import estimate_pdf
from .infer import infer
from .mesh import mesh
from .plot_mesh import plot_mesh
from .unmesh import unmesh


def infer_assuming_independence(
    element_x_dimension, value, plot=True, names=None,
):

    n_dimension = element_x_dimension.shape[1]

    if names is None:

        names = tuple("Dimension {} Variable".format(i) for i in range(n_dimension))

    (
        target_mesh_grid_point_x_dimension,
        target_mesh_grid_point_posterior_probability,
    ) = estimate_pdf(element_x_dimension[:, -1:], plot=plot, names=names[-1:],)

    target_dimensino_grids, target_probability = unmesh(
        target_mesh_grid_point_x_dimension, target_mesh_grid_point_posterior_probability
    )

    target_dimension_grid = target_dimensino_grids[0]

    target_value_index = absolute(target_dimension_grid - value).argmin()

    infer_returns = tuple(
        infer(
            element_x_dimension[:, [i, -1]],
            value,
            plot=plot,
            names=(names[i], names[-1]),
        )
        for i in range(n_dimension - 1)
    )

    no_target_mesh_grid_point_x_dimension = mesh(
        tuple(infer_return[0] for infer_return in infer_returns)
    )

    no_target_mesh_grid_point_posterior_probability = apply_along_axis(
        product, 1, mesh(tuple(infer_return[1] for infer_return in infer_returns)),
    ) / (target_probability[target_value_index] ** (n_dimension - 2))

    if plot:

        plot_mesh(
            no_target_mesh_grid_point_x_dimension,
            no_target_mesh_grid_point_posterior_probability,
            names=names,
            value_name="P({} = {:.2e} (~{}) | {})".format(
                names[-1],
                target_dimension_grid[target_value_index],
                value,
                *names[:-1],
            ),
        )

    return (
        no_target_mesh_grid_point_x_dimension,
        no_target_mesh_grid_point_posterior_probability,
    )
