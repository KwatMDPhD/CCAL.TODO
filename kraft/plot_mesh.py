from pandas import DataFrame, Index

from .plot_heat_map import plot_heat_map
from .plot_plotly import plot_plotly
from .unmesh import unmesh


def plot_mesh(
    point_x_dimension, value, names=None, value_name="Value",
):

    n_dimension = point_x_dimension.shape[1]

    if names is None:

        names = tuple("Dimension {}".format(i) for i in range(n_dimension))

    grids, value = unmesh(point_x_dimension, value)

    if n_dimension == 1:

        plot_plotly(
            {
                "layout": {
                    "xaxis": {"title": {"text": names[0]}},
                    "yaxis": {"title": {"text": value_name}},
                },
                "data": [{"type": "scatter", "x": grids[0], "y": value}],
            },
            None,
        )

    elif n_dimension == 2:

        plot_heat_map(
            DataFrame(
                value,
                index=Index(("{:.2e} *".format(i) for i in grids[0]), name=names[0]),
                columns=Index(("* {:.2e}".format(i) for i in grids[1]), name=names[1]),
            ),
            layout={"title": {"text": value_name}},
        )

    else:

        print("======== {} ========".format(value_name))

        print("Grids:")

        print(grids)

        print("Value (unmesh):")

        print("Shape: {}".format(value.shape))

        print("Min: {:.2e}".format(value.min()))

        print("Max: {:.2e}".format(value.max()))
