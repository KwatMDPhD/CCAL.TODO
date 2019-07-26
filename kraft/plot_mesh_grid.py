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

    dimension_grids, value = unmesh(mesh_grid_point_x_dimension, mesh_grid_point_value)

    if dimension_names is None:

        dimension_names = tuple(f"Dimension {i} Variable" for i in range(n_dimension))

    if n_dimension == 1:

        plot_plotly_figure(
            {
                "layout": {
                    "title": {"text": title_text},
                    "xaxis": {
                        "title": {
                            "text": f"{dimension_names[0]} (n={dimension_grids[0].size})"
                        }
                    },
                    "yaxis": {"title": {"text": "Value"}},
                },
                "data": [
                    {
                        "type": "scatter",
                        "x": dimension_grids[0],
                        "y": value,
                        "mode": "markers",
                        "marker": {"color": "#20d9ba"},
                    }
                ],
            },
            None,
        )

    elif n_dimension == 2:

        plot_heat_map(
            DataFrame(
                value,
                index=(f"{i:.3f} *" for i in dimension_grids[0]),
                columns=(f"* {i:.3f}" for i in dimension_grids[1]),
            ),
            title_text=title_text,
            xaxis_title_text=dimension_names[1],
            yaxis_title_text=dimension_names[0],
        )

    else:

        print("=" * 80)

        print(f"{dimension_grids.shape[0]} Dimension Grids:")

        print(dimension_grids)

        print(f"Value Shape = {value.shape}")

        print(f"Minimum Value = {value.min()}")

        print(f"Maximum Value = {value.max()}")

        print("^" * 80)
