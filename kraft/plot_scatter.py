from pandas import Series

from .plot_plotly_figure import plot_plotly_figure


def plot_scatter(coordinates_s, layout=None):

    data = []

    for i, coordinates in enumerate(coordinates_s):

        if len(coordinates) == 1:

            y = coordinates[0]

            x = tuple(range(len(y)))

        else:

            x, y = coordinates

        if isinstance(y, Series):

            name = y.name

            text = y.index

        y_size = len(y)

        if y_size < 64:

            mode = "markers+text"

        elif y_size < 256:

            mode = "markers"

        else:

            mode = "lines"

        data.append(
            {
                "type": "scatter",
                "name": name,
                "x": x,
                "y": y,
                "text": text,
                "mode": mode,
                "marker": {"size": 3.2},
            }
        )

    if layout is None:

        layout = {}

    plot_plotly_figure({"layout": layout, "data": data})
