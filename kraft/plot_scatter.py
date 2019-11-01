from pandas import Series

from .plot_plotly_figure import plot_plotly_figure


def plot_scatter(coordinates_s, traces=(), layout=None):

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

        else:

            name = None

            text = None

        if 0 < len(traces):

            trace_template = traces[i]

        else:

            y_size = len(y)

            if y_size < 64:

                mode = "markers+text"

            elif y_size < 256:

                mode = "markers"

            else:

                mode = "lines"

            trace_template = {
                "mode": mode,
            }

        data.append(
            {
                "type": "scatter",
                "name": name,
                "x": x,
                "y": y,
                "text": text,
                **trace_template,
            },
        )

    if layout is None:

        layout = {}

    plot_plotly_figure({"layout": layout, "data": data})
