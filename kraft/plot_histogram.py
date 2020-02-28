from pandas import Series

from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .get_color import get_color
from .merge_2_dicts import merge_2_dicts
from .plot_plotly import plot_plotly


def plot_histogram(
    xs, histnorm=None, bin_size=None, plot_rug=None, layout=None, html_file_path=None
):

    if histnorm is None:

        yaxis2_title_text = "N"

    else:

        yaxis2_title_text = histnorm.title()

    if plot_rug is None:

        plot_rug = all(len(x) < 1e3 for x in xs)

    n_x = len(xs)

    if plot_rug:

        rug_height = 0.04

        yaxis_domain_max = n_x * rug_height

        yaxis2_domain_min = yaxis_domain_max + rug_height

    else:

        yaxis_domain_max = 0

        yaxis2_domain_min = 0

    yaxis_domain = 0, yaxis_domain_max

    yaxis2_domain = yaxis2_domain_min, 1

    layout_ = {
        "xaxis": {"anchor": "y"},
        "yaxis": {
            "domain": yaxis_domain,
            "zeroline": False,
            "dtick": 1,
            "showticklabels": False,
        },
        "yaxis2": {"domain": yaxis2_domain, "title": {"text": yaxis2_title_text}},
        "showlegend": True,
    }

    if layout is None:

        layout = layout_

    else:

        layout = merge_2_dicts(layout_, layout)

    data = []

    for x_index, x in enumerate(xs):

        if isinstance(x, Series):

            name = x.name

            text = x.index

        else:

            name = x_index

            text = None

        color = get_color(DATA_TYPE_COLORSCALE["categorical"], x_index, n_x)

        data.append(
            {
                "yaxis": "y2",
                "type": "histogram",
                "legendgroup": x_index,
                "name": name,
                "x": x,
                "histnorm": histnorm,
                "xbins": {"size": bin_size},
                "marker": {"color": color},
            }
        )

        if plot_rug:

            data.append(
                {
                    "legendgroup": x_index,
                    "name": name,
                    "showlegend": False,
                    "x": x,
                    "y": (x_index,) * len(x),
                    "text": text,
                    "mode": "markers",
                    "marker": {"symbol": "line-ns-open", "color": color},
                    "hoverinfo": "x+text",
                }
            )

    plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)
