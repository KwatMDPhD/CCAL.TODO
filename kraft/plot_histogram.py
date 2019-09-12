from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .get_colorscale_color import get_colorscale_color
from .merge_2_dicts_recursively import merge_2_dicts_recursively
from .plot_plotly_figure import plot_plotly_figure


def plot_histogram(serieses, histnorm=None, layout=None, html_file_path=None):

    if histnorm is None:

        yaxis2_title_text = "N"

    else:

        yaxis2_title_text = histnorm.title()

    layout_template = {
        "xaxis": {"anchor": "y"},
        "yaxis": {"domain": (0, 0.1), "dtick": 1, "showticklabels": False},
        "yaxis2": {"domain": (0.15, 1), "title": {"text": yaxis2_title_text}},
    }

    if layout is None:

        layout = layout_template

    else:

        layout = merge_2_dicts_recursively(layout_template, layout)

    data = []

    for i, series in enumerate(serieses):

        color = get_colorscale_color(
            DATA_TYPE_COLORSCALE["categorical"], i / (len(serieses) - 1)
        )

        data.append(
            {
                "yaxis": "y2",
                "type": "histogram",
                "legendgroup": series.name,
                "name": series.name,
                "x": series,
                "histnorm": histnorm,
                "marker": {"color": color},
            }
        )

        data.append(
            {
                "type": "scatter",
                "legendgroup": series.name,
                "showlegend": False,
                "x": series,
                "y": (0,) * series.size,
                "text": series.index,
                "mode": "markers",
                "marker": {"symbol": "line-ns-open", "color": color},
            }
        )

    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)
