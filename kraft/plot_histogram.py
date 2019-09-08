from numpy import arange

from .pick_colors import pick_colors
from .plot_plotly_figure import plot_plotly_figure


def plot_histogram(
    serieses, histnorm="", plot_rug=None, layout=None, xaxis=None, html_file_path=None
):

    if plot_rug is None:

        plot_rug = all(series.size < 1e3 for series in serieses)

    if plot_rug:

        yaxis_max = 0.16

        yaxis2_min = yaxis_max + 0.08

    else:

        yaxis_max = 0

        yaxis2_min = 0

    data = []

    colors = pick_colors(arange(len(serieses)))

    for i, series in enumerate(serieses):

        color = colors[i]

        data.append(
            {
                "yaxis": "y2",
                "type": "histogram",
                "name": series.name,
                "legendgroup": series.name,
                "x": series,
                "histnorm": histnorm,
                "marker": {"color": color},
                "opacity": 0.8,
            }
        )

        if plot_rug:

            data.append(
                {
                    "type": "scatter",
                    "legendgroup": series.name,
                    "showlegend": False,
                    "x": series,
                    "y": (i,) * series.size,
                    "text": series.index,
                    "mode": "markers",
                    "marker": {"symbol": "line-ns-open", "color": color},
                }
            )

    plot_plotly_figure(
        {
            "layout": {
                "xaxis": {"anchor": "y", **xaxis},
                "yaxis": {
                    "domain": (0, yaxis_max),
                    "dtick": 1,
                    "zeroline": False,
                    "showticklabels": False,
                },
                "yaxis2": {
                    "domain": (yaxis2_min, 1),
                    "title": {"text": histnorm.title()},
                },
                "barmode": "overlay",
                **layout,
            },
            "data": data,
        },
        html_file_path,
    )
