from ..dictionary import merge
from .CATEGORICAL_COLORSCALE import CATEGORICAL_COLORSCALE
from .get_color import get_color
from .plot_plotly import plot_plotly


def plot_histogram(
    se_,
    no=None,
    xbins_size=None,
    colorscale=CATEGORICAL_COLORSCALE,
    layout=None,
    pa="",
):

    ru = all(se.size <= 1e3 for se in se_)

    n_tr = len(se_)

    if ru:

        he = 0.04

        ma = n_tr * he

        mi = ma + he

    else:

        ma = 0

        mi = 0

    if no is None:

        yaxis2_title = "N"

    else:

        yaxis2_title = no.title()

    if layout is None:

        layout = {}

    layout = merge(
        {
            "xaxis": {
                "anchor": "y",
            },
            "yaxis": {
                "domain": [0, ma],
                "zeroline": False,
                "dtick": 1,
                "showticklabels": False,
            },
            "yaxis2": {
                "domain": [mi, 1],
                "title": yaxis2_title,
            },
        },
        layout,
    )

    data = []

    for ie, se in enumerate(se_):

        co = get_color(colorscale, ie / max(1, (n_tr - 1)))

        trace = {
            "legendgroup": ie,
            "name": se.name,
            "x": se.values,
        }

        data.append(
            {
                "yaxis": "y2",
                "type": "histogram",
                "histnorm": no,
                "xbins": {
                    "size": xbins_size,
                },
                "marker": {
                    "color": co,
                },
                **trace,
            }
        )

        if ru:

            data.append(
                {
                    "showlegend": False,
                    "y": [ie] * se.size,
                    "text": se.index,
                    "mode": "markers",
                    "marker": {
                        "symbol": "line-ns-open",
                        "color": co,
                    },
                    "hoverinfo": "x+text",
                    **trace,
                }
            )

    plot_plotly(
        {
            "data": data,
            "layout": layout,
        },
        pa=pa,
    )
