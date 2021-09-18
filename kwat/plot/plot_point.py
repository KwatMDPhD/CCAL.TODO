from ..dictionary import merge
from .plot_plotly import plot_plotly


def plot_point(da, layout=None, pa=""):

    co_ = da.columns.values

    co1, co2 = co_[:2]

    data = [
        {
            "name": "Point",
            "y": da.loc[:, co1],
            "x": da.loc[:, co2],
            "text": da.index.values,
            "mode": "markers",
            "marker": {
                "size": da.loc[:, "Size"],
                "color": da.loc[:, "Color"],
                "opacity": da.loc[:, "Opacity"],
                "line": {
                    "width": 0,
                },
            },
        }
    ]

    annotations = []

    if "Annotate" in co_:

        for text, (y, x) in da.loc[da.loc[:, "Annotate"], [co1, co2]].iterrows():

            annotations.append(
                {
                    "y": y,
                    "x": x,
                    "text": text,
                    "font": {
                        "size": 8,
                    },
                    "arrowhead": 2,
                    "arrowsize": 1,
                    "clicktoshow": "onoff",
                }
            )

    if layout is None:

        layout = {}

    layout = merge(
        {
            "title": "Plot Point",
            "yaxis": {
                "title": co1,
            },
            "xaxis": {
                "title": co2,
            },
            "annotations": annotations,
        },
        layout,
    )

    plot_plotly(
        {
            "data": data,
            "layout": layout,
        },
        pa=pa,
    )
