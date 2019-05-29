from pandas import DataFrame

from .plot_heat_map import plot_heat_map
from .unmesh import unmesh


def plot_2d_mesh_grid(
    mesh_grid_point_x_dimension,
    mesh_grid_point_value,
    title_template="f({}, {})",
    dimension_names=None,
):

    (dimension_0_grid, dimension_1_grid), _2d_array = unmesh(
        mesh_grid_point_x_dimension, mesh_grid_point_value
    )

    if dimension_names is None:

        dimension_names = ("Dimension 0 Variable", "Dimension 1 Variable")

    plot_heat_map(
        DataFrame(
            _2d_array,
            index=(f"*{i:.3f}" for i in dimension_0_grid),
            columns=(f"*{i:.3f}" for i in dimension_1_grid),
        ),
        title=title_template.format(*dimension_names),
        xaxis_title=dimension_names[1],
        yaxis_title=dimension_names[0],
    )
