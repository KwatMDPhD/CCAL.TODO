from .plot_plotly_figure import plot_plotly_figure


def plot_histogram(
    serieses, histnorm="", plot_rug=None, layout=None, xaxis=None, html_file_path=None
):

    xaxis_template = {"anchor": "y"}

    if xaxis is None:

        xaxis = xaxis_template

    else:

        xaxis = {**xaxis_template, **xaxis}

    if plot_rug is None:

        plot_rug = all(series.size < 1e3 for series in serieses)

    if plot_rug:

        yaxis_max = 0.16

        yaxis2_min = yaxis_max + 0.08

    else:

        yaxis_max = 0

        yaxis2_min = 0

    layout_template = {
        "xaxis": xaxis,
        "yaxis": {
            "domain": (0, yaxis_max),
            "dtick": 1,
            "zeroline": False,
            "showticklabels": False,
        },
        "yaxis2": {"domain": (yaxis2_min, 1), "title": {"text": histnorm.title()}},
        "barmode": "overlay",
    }

    if layout is None:

        layout = layout_template

    else:

        layout = {**layout_template, **layout}

    data = []

    for i, series in enumerate(serieses):

        data.append(
            {
                "yaxis": "y2",
                "type": "histogram",
                "name": series.name,
                "legendgroup": series.name,
                "x": series,
                "histnorm": histnorm,
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
                    "marker": {"symbol": "line-ns-open"},
                }
            )

    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)
