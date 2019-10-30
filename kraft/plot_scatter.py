from .plot_plotly_figure import plot_plotly_figure
from pandas import Series


def plot_scatter(xs, ys, layout=None):

    if xs is None:

        xs = tuple(tuple(range(len(y))) for y in ys)

    elif len(xs) == 1:

        xs = xs * len(ys)

    names = []

    texts = []

    for y in ys:

        if isinstance(y, Series):

            names.append(y.name)

            texts.append(y.index)

        else:

            names.append(None)

            texts.append(None)

    data = [
        {
            "type": "scatter",
            "name": name,
            "x": x,
            "y": y,
            "text": text,
            "mode": "lines",
            "marker": {"size": 1},
        }
        for name, x, y, text in zip(names, xs, ys, texts)
    ]

    if layout is None:

        layout = {}

    plot_plotly_figure({"layout": layout, "data": data})

