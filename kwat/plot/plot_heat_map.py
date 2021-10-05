from numpy import argsort, nonzero, unique

from ..dictionary import merge
from .COLORBAR import COLORBAR
from .NAME_COLORSCALE import NAME_COLORSCALE
from .plot_plotly import plot_plotly


def _get_center_index(gr_, gr):

    ie1, ie2 = nonzero(gr_ == gr)[0][[0, -1]]

    return ie1 + (ie2 - ie1) / 2


def plot_heat_map(
    da,
    colorscale=NAME_COLORSCALE["continuous"],
    gr1_=(),
    gr2_=(),
    colorscale1=NAME_COLORSCALE["categorical"],
    colorscale2=NAME_COLORSCALE["categorical"],
    gr1_la=None,
    gr2_la=None,
    layout=None,
    annotation1=None,
    annotation2=None,
    pr="",
):

    if 0 < len(gr1_):

        ie_ = argsort(gr1_)

        gr1_ = gr1_[ie_]

        da = da.iloc[ie_, :]

    if 0 < len(gr2_):

        ie_ = argsort(gr2_)

        gr2_ = gr2_[ie_]

        da = da.iloc[:, ie_]

    domain = [0, 0.95]

    if layout is None:

        layout = {}

    axis = {
        "domain": [0.96, 1],
        "showgrid": False,
        "showline": False,
        "zeroline": False,
        "showticklabels": False,
    }

    layout = merge(
        {
            "title": {"text": "Heat Map"},
            "yaxis": {
                "title": {"text": "{} (n={})".format(da.index.name, da.shape[0])},
                "domain": domain,
            },
            "xaxis": {
                "title": {"text": "{} (n={})".format(da.columns.name, da.shape[1])},
                "domain": domain,
            },
            "yaxis2": axis,
            "xaxis2": axis,
            "annotations": [],
        },
        layout,
    )

    colorbar_x = 1.04

    data = [
        {
            "type": "heatmap",
            "z": da.values[::-1],
            "y": da.index.values[::-1],
            "x": da.columns.values,
            "colorscale": colorscale,
            "colorbar": merge(COLORBAR, {"x": colorbar_x}),
        }
    ]

    heatmap = {"type": "heatmap", "colorbar": merge(COLORBAR, {"dtick": 1})}

    annotation = {"showarrow": False}

    if 0 < len(gr1_):

        gr1_ = gr1_[::-1]

        colorbar_x += 0.1

        data.append(
            merge(
                heatmap,
                {
                    "xaxis": "x2",
                    "z": gr1_.reshape([-1, 1]),
                    "colorscale": colorscale1,
                    "colorbar": {"x": colorbar_x},
                    "hoverinfo": "z+y",
                },
            )
        )

        if gr1_la is not None:

            if annotation1 is None:

                annotation1 = {}

            layout["annotations"] += [
                merge(
                    merge(
                        annotation,
                        {
                            "xref": "x2",
                            "x": 0,
                            "xanchor": "left",
                            "y": _get_center_index(gr1_, gr),
                            "text": gr1_la[gr],
                        },
                    ),
                    annotation1,
                )
                for gr in unique(gr1_)
            ]

    if 0 < len(gr2_):

        colorbar_x += 0.1

        data.append(
            merge(
                heatmap,
                {
                    "yaxis": "y2",
                    "z": gr2_.reshape([1, -1]),
                    "colorscale": colorscale2,
                    "colorbar": {"x": colorbar_x},
                    "hoverinfo": "z+x",
                },
            )
        )

        if gr2_la is not None:

            if annotation2 is None:

                annotation2 = {}

            layout["annotations"] += [
                merge(
                    merge(
                        annotation,
                        {
                            "yref": "y2",
                            "y": 0,
                            "yanchor": "bottom",
                            "textangle": -90,
                            "x": _get_center_index(gr2_, gr),
                            "text": gr2_la[gr],
                        },
                    ),
                    annotation2,
                )
                for gr in unique(gr2_)
            ]

    plot_plotly(data, layout, pr=pr)
