from pandas import DataFrame

from .plot_heat_map import plot_heat_map
from .plot_plotly_figure import plot_plotly_figure
from .unmesh import unmesh


def plot_mesh_grid(
    mesh_grid_point_x_dimension,
    mesh_grid_point_value,
    title_text=None,
    dimension_names=None,
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

        dimension_0_name = dimension_names[0]

        dimension_0_grid_size = dimension_grids[0].size

        plot_plotly_figure(
            {
                "layout": {
                    "title": {"text": title_text},
                    "xaxis": {
                        "title": {
                            "text": "{} (n={})".format(
                                dimension_0_name, dimension_0_grid_size
                            )
                        }
                    },
                    "yaxis": {"title": {"text": "Value"}},
                },
                "data": [
                    {
                        "type": "scatter",
                        "x": dimension_grids[0],
                        "y": value_reshaped,
                        "mode": "markers",
                    }
                ],
            },
            None,
        )

    elif n_dimension == 2:

        plot_heat_map(
            DataFrame(
                value_reshaped,
                index=("{:.3f} *".format(i) for i in dimension_grids[0]),
                columns=("* {:.3f}".format(i) for i in dimension_grids[1]),
            ),
            title_text=title_text,
            xaxis_title_text=dimension_names[1],
            yaxis_title_text=dimension_names[0],
        )

    else:

        print("=" * 80)

        print("N dimension: {}".format(n_dimension))

        print("Dimension grids:")

        print(dimension_grids)

        print("Value shape: {}".format(value_reshaped.shape))

        print("Value min: {}".format(value_reshaped.min()))

        print("Value max: {}".format(value_reshaped.max()))

        print("^" * 80)
