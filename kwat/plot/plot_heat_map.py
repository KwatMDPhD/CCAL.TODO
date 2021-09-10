from numpy import argsort, unique

from ..dictionary import merge
from .categorical_colorscale import categorical_colorscale
from .colorbar import colorbar
from .continuous_colorscale import continuous_colorscale
from .plot_plotly import plot_plotly


def plot_heat_map(
    nu_an_an,
    colorscale=continuous_colorscale,
    gr1_=(),
    gr2_=(),
    colorscale1=categorical_colorscale,
    colorscale2=categorical_colorscale,
    gr1_la=None,
    gr2_la=None,
    layout=None,
    annotation1=None,
    annotation2=None,
    pa="",
):

    if 0 < len(gr1_):

        ie_ = argsort(gr1_)

        gr1_ = gr1_[ie_]

        nu_an_an = nu_an_an.iloc[ie_, :]

    if 0 < len(gr2_):

        ie_ = argsort(gr2_)

        gr2_ = gr2_[ie_]

        nu_an_an = nu_an_an.iloc[:, ie_]

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
            "yaxis": {
                "title": "{} (n={})".format(nu_an_an.index.name, nu_an_an.shape[0]),
                "domain": domain,
            },
            "xaxis": {
                "title": "{} (n={})".format(nu_an_an.columns.name, nu_an_an.shape[1]),
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
            "z": nu_an_an.values[::-1],
            "y": nu_an_an.index.values[::-1],
            "x": nu_an_an.columns.values,
            "colorscale": colorscale,
            "colorbar": {
                **colorbar,
                "x": colorbar_x,
            },
        }
    ]

    if 0 < len(gr1_):

        gr1_ = gr1_[::-1]

        colorbar_x += 0.1

        data.append(
            {
                "xaxis": "x2",
                "type": "heatmap",
                "z": gr1_.reshape([-1, 1]),
                "colorscale": colorscale1,
                "colorbar": {
                    **colorbar,
                    "x": colorbar_x,
                    "dtick": 1,
                },
                "hoverinfo": "z+y",
            }
        )

        if gr1_la is not None:

            if annotation1 is None:

                annotation1 = {}

            layout["annotations"] += [
                merge(
                    {
                        "xref": "x2",
                        "x": 0,
                        "xanchor": "left",
                        "showarrow": False,
                        "y": _get_center_index(gr1_, gr),
                        "text": gr1_la[gr],
                    },
                    annotation1,
                )
                for gr in unique(gr1_)
            ]

    if 0 < len(gr2_):

        colorbar_x += 0.1

        data.append(
            {
                "yaxis": "y2",
                "type": "heatmap",
                "z": gr2_.reshape([1, -1]),
                "colorscale": colorscale2,
                "colorbar": {
                    **colorbar,
                    "x": colorbar_x,
                    "dtick": 1,
                },
                "hoverinfo": "z+x",
            }
        )

        if gr2_la is not None:

            if annotation2 is None:

                annotation2 = {}

            layout["annotations"] += [
                merge(
                    {
                        "yref": "y2",
                        "y": 0,
                        "yanchor": "bottom",
                        "textangle": -90,
                        "showarrow": False,
                        "x": _get_center_index(gr2_, gr),
                        "text": gr2_la[gr],
                    },
                    annotation2,
                )
                for gr in unique(gr2_)
            ]

    plot_plotly(
        {
            "data": data,
            "layout": layout,
        },
        pa=pa,
    )
