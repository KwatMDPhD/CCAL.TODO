from pandas import DataFrame, Index

from ..plot.plot_heat_map import plot_heat_map
from ..plot.plot_plotly import plot_plotly
from .unmesh import unmesh


def plot_mesh(
    point_x_dimension, value, names=None, value_name="Value", html_file_path=None
):

    n_dimension = point_x_dimension.shape[1]

    if names is None:

        names = tuple("Dimension {}".format(i) for i in range(n_dimension))

    grids, value = unmesh(point_x_dimension, value)

    for grid_index, grid in enumerate(grids):

        print(
            "Dimension {} grid: size={} min={:.2e} max={:.2e}".format(
                grid_index, grid.size, grid.min(), grid.max()
            )
        )

    print("Value: min={:.2e} max={:.2e}".format(value.min(), value.max()))

    if n_dimension == 1:

        plot_plotly(
            {
                "layout": {
                    "xaxis": {"title": {"text": names[0]}},
                    "yaxis": {"title": {"text": value_name}},
                },
                "data": [
                    {"type": "scatter", "x": grids[0], "y": value, "mode": "lines"}
                ],
            },
            html_file_path=html_file_path,
        )

    elif n_dimension == 2:

        plot_heat_map(
            DataFrame(
                value,
                index=Index(("{:.2e} *".format(i) for i in grids[0]), name=names[0]),
                columns=Index(("* {:.2e}".format(i) for i in grids[1]), name=names[1]),
            ),
            layout={"title": {"text": value_name}},
            html_file_path=html_file_path,
        )
