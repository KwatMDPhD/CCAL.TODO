from pandas import Series

from .plot_plotly_figure import plot_plotly_figure


def plot_scatter(xs, ys, layout=None):

    if xs is None:

        xs = tuple(tuple(range(len(y))) for y in ys)

    elif len(xs) == 1:

        xs = xs * len(ys)

    names = []

    texts = []

    modes = []

    for y in ys:

        if isinstance(y, Series):

            names.append(y.name)

            texts.append(y.index)

        else:

            names.append(None)

            texts.append(None)

        y_size = len(y)

        if y_size < 64:

            modes.append("markers+text")

        elif y_size < 256:

            modes.append("markers")

        else:

            modes.append("lines")

    data = [
        {
            "type": "scatter",
            "name": name,
            "x": x,
            "y": y,
            "text": text,
            "mode": mode,
            "marker": {"size": 3.2},
        }
        for name, x, y, text, mode in zip(names, xs, ys, texts, modes)
    ]

    if layout is None:

        layout = {}

    plot_plotly_figure({"layout": layout, "data": data})
