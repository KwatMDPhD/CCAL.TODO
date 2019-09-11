from pandas import DataFrame

from .plot_heat_map import plot_heat_map
from .plot_plotly_figure import plot_plotly_figure
from .unmesh import unmesh


def plot_mesh_grid(
    mesh_grid_point_x_dimension,
    mesh_grid_point_value,
    dimension_names=None,
    value_name="Value",
):

    n_dimension = mesh_grid_point_x_dimension.shape[1]

    if dimension_names is None:

        dimension_names = tuple(
            "Dimension {} Variable".format(i) for i in range(n_dimension)
        )

    dimension_grids, value_reshaped = unmesh(
        mesh_grid_point_x_dimension, mesh_grid_point_value
    )

    if n_dimension == 1:

        plot_plotly_figure(
            {
                "layout": {
                    "xaxis": {"title": {"text": dimension_names[0]}},
                    "yaxis": {"title": {"text": value_name}},
                },
                "data": [
                    {"type": "scatter", "x": dimension_grids[0], "y": value_reshaped}
                ],
            },
            None,
        )

    elif n_dimension == 2:

        plot_heat_map(
            DataFrame(
                value_reshaped,
                index=("{:.2e} *".format(i) for i in dimension_grids[0]),
                columns=("* {:.2e}".format(i) for i in dimension_grids[1]),
            ),
            layout={
                "title": {"text": value_name},
                "xaxis": {"title": {"text": dimension_names[1]}},
                "yaxis": {"title": {"text": dimension_names[0]}},
            },
        )

    else:

        print("======== {} ========".format(value_name))

        print("Dimension Grids:")

        print(dimension_grids)

        print("Value Reshaped:")

        print("Shape: {}".format(value_reshaped.shape))

        print("Min: {:.2e}".format(value_reshaped.min()))

        print("Max: {:.2e}".format(value_reshaped.max()))
