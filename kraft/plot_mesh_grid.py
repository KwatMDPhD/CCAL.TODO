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

        dimension_names = tuple(f"Dimension {i} Variable" for i in range(n_dimension))

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
                            "text": f"{dimension_0_name} (n={dimension_0_grid_size})"
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
                index=(f"{i:.3f} *" for i in dimension_grids[0]),
                columns=(f"* {i:.3f}" for i in dimension_grids[1]),
            ),
            title_text=title_text,
            xaxis_title_text=dimension_names[1],
            yaxis_title_text=dimension_names[0],
        )

    else:

        print("=" * 80)

        print("N Dimension:")

        print(n_dimension)

        print("Dimension Grids:")

        print(dimension_grids)

        print("Value Shape, Min, and Max:")

        print(value_reshaped.shape, value_reshaped.min(), value_reshaped.max())

        print("^" * 80)
