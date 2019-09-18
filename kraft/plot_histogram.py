from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .get_colorscale_color import get_colorscale_color
from .merge_2_dicts_recursively import merge_2_dicts_recursively
from .plot_plotly_figure import plot_plotly_figure


def plot_histogram(
    serieses, histnorm=None, plot_rug=True, layout=None, html_file_path=None
):

    if histnorm is None:

        yaxis2_title_text = "N"

    else:

        yaxis2_title_text = histnorm.title()

    if plot_rug:

        yaxis_domain = (0, 0.1)

        yaxis2_domain = (0.15, 1)

    else:

        yaxis_domain = (0, 0)

        yaxis2_domain = (0, 1)

    layout_template = {
        "xaxis": {"anchor": "y"},
        "yaxis": {"domain": yaxis_domain, "dtick": 1, "showticklabels": False},
        "yaxis2": {"domain": yaxis2_domain, "title": {"text": yaxis2_title_text}},
    }

    if layout is None:

        layout = layout_template

    else:

        layout = merge_2_dicts_recursively(layout_template, layout)

    data = []

    for i, series in enumerate(serieses):

        color = get_colorscale_color(
            DATA_TYPE_COLORSCALE["categorical"], i, len(serieses)
        )

        data.append(
            {
                "yaxis": "y2",
                "type": "histogram",
                "legendgroup": i,
                "name": series.name,
                "x": series,
                "histnorm": histnorm,
                "marker": {"color": color},
            }
        )

        if plot_rug:

            data.append(
                {
                    "type": "scatter",
                    "legendgroup": i,
                    "showlegend": False,
                    "x": series,
                    "y": (0,) * series.size,
                    "text": series.index,
                    "mode": "markers",
                    "marker": {"symbol": "line-ns-open", "color": color},
                    "hoverinfo": "x+text",
                }
            )

    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)
