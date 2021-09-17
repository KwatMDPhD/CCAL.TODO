from numpy import argsort, nonzero, unique

from ..dictionary import merge
from .CATEGORICAL_COLORSCALE import CATEGORICAL_COLORSCALE
from .COLORBAR_TEMPLATE import COLORBAR_TEMPLATE
from .CONTINUOUS_COLORSCALE import CONTINUOUS_COLORSCALE
from .plot_plotly import plot_plotly


def _get_center_index(gr_, gr):

    ie1, ie2 = nonzero(gr_ == gr)[0][[0, -1]]

    return ie1 + (ie2 - ie1) / 2


def plot_heat_map(
    nu_an_an,
    colorscale=CONTINUOUS_COLORSCALE,
    gr1_=(),
    gr2_=(),
    colorscale1=CATEGORICAL_COLORSCALE,
    colorscale2=CATEGORICAL_COLORSCALE,
    gr1_la=None,
    gr2_la=None,
    layout=None,
    ANNOTATION_TEMPLATE1=None,
    ANNOTATION_TEMPLATE2=None,
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
                "title": "{} (n={})".format(nu_an_an.COLUMNS.name, nu_an_an.shape[1]),
                "domain": domain,
            },
            "yaxis2": axis,
            "xaxis2": axis,
            "ANNOTATION_TEMPLATEs": [],
        },
        layout,
    )

    colorbar_x = 1.04

    data = [
        {
            "type": "HEATMAP_TEMPLATE",
            "z": nu_an_an.values[::-1],
            "y": nu_an_an.index.values[::-1],
            "x": nu_an_an.COLUMNS.values,
            "colorscale": colorscale,
            "colorbar": {
                **COLORBAR_TEMPLATE,
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
                "type": "HEATMAP_TEMPLATE",
                "z": gr1_.reshape([-1, 1]),
                "colorscale": colorscale1,
                "colorbar": {
                    **COLORBAR_TEMPLATE,
                    "x": colorbar_x,
                    "dtick": 1,
                },
                "hoverinfo": "z+y",
            }
        )

        if gr1_la is not None:

            if ANNOTATION_TEMPLATE1 is None:

                ANNOTATION_TEMPLATE1 = {}

            layout["ANNOTATION_TEMPLATEs"] += [
                merge(
                    {
                        "xref": "x2",
                        "x": 0,
                        "xanchor": "left",
                        "showarrow": False,
                        "y": _get_center_index(gr1_, gr),
                        "text": gr1_la[gr],
                    },
                    ANNOTATION_TEMPLATE1,
                )
                for gr in unique(gr1_)
            ]

    if 0 < len(gr2_):

        colorbar_x += 0.1

        data.append(
            {
                "yaxis": "y2",
                "type": "HEATMAP_TEMPLATE",
                "z": gr2_.reshape([1, -1]),
                "colorscale": colorscale2,
                "colorbar": {
                    **COLORBAR_TEMPLATE,
                    "x": colorbar_x,
                    "dtick": 1,
                },
                "hoverinfo": "z+x",
            }
        )

        if gr2_la is not None:

            if ANNOTATION_TEMPLATE2 is None:

                ANNOTATION_TEMPLATE2 = {}

            layout["ANNOTATION_TEMPLATEs"] += [
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
                    ANNOTATION_TEMPLATE2,
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
