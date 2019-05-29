from pandas import DataFrame

from .plot_heat_map import plot_heat_map
from .unmesh import unmesh


def plot_2d_mesh_grid(
    mesh_grid_point_x_dimension,
    mesh_grid_point_value,
    title_template="f({}, {})",
    names=None,
):

    (dimension_0_grid, dimension_1_grid), _2d_array = unmesh(
        mesh_grid_point_x_dimension, mesh_grid_point_value
    )

    if names is None:

        names = ("Variable 0", "Variable 1")

    plot_heat_map(
        DataFrame(
            _2d_array,
            index=(f"*{i:.3f}" for i in dimension_0_grid),
            columns=(f"*{i:.3f}" for i in dimension_1_grid),
        ),
        title=title_template.format(*names),
        xaxis_title=names[1],
        yaxis_title=names[0],
    )
