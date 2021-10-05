from ..dictionary import merge
from .get_color import get_color
from .NAME_COLORSCALE import NAME_COLORSCALE
from .plot_plotly import plot_plotly


def plot_histogram(
    se_,
    no=None,
    xbins_size=None,
    colorscale=NAME_COLORSCALE["categorical"],
    layout=None,
    pr="",
):

    ru = all(se.size <= 1e5 for se in se_)

    n_se = len(se_)

    if ru:

        he = 0.04

        fr = min(n_se * he, 0.5)

    else:

        fr = 0

    if no is None:

        yaxis2_title = "N"

    else:

        yaxis2_title = no.title()

    if layout is None:

        layout = {}

    layout = merge(
        {
            "xaxis": {"anchor": "y"},
            "yaxis": {
                "domain": [0, fr],
                "zeroline": False,
                "dtick": 1,
                "showticklabels": False,
            },
            "yaxis2": {"domain": [fr, 1], "title": {"text": yaxis2_title}},
        },
        layout,
    )

    data = []

    for ie, se in enumerate(se_):

        co = get_color(colorscale, ie, ex_=[0, n_se - 1])

        trace = {"legendgroup": ie, "name": se.name, "x": se.values}

        data.append(
            merge(
                trace,
                {
                    "yaxis": "y2",
                    "type": "histogram",
                    "histnorm": no,
                    "xbins": {"size": xbins_size},
                    "marker": {"color": co},
                },
            )
        )

        if ru:

            data.append(
                merge(
                    trace,
                    {
                        "showlegend": False,
                        "y": [ie] * se.size,
                        "text": se.index.values,
                        "mode": "markers",
                        "marker": {"symbol": "line-ns-open", "color": co},
                        "hoverinfo": "x+text",
                    },
                )
            )

    plot_plotly({"data": data, "layout": layout}, pr=pr)
