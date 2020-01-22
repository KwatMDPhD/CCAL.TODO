from numpy import arange

from .merge_2_dicts import merge_2_dicts
from .plot_plotly import plot_plotly


def plot_errors(errors, layout=None, html_file_path=None):

    layout_template = {
        "xaxis": {"title": "Iteration"},
        "yaxis": {"title": "Error"},
        "annotations": [
            {"x": errors_.size, "y": errors_[-1], "text": "{:.2e}".format(errors_[-1])}
            for i, errors_ in enumerate(errors)
        ],
    }

    if layout is None:

        layout = layout_template

    else:

        layout = merge_2_dicts(layout_template, layout)

    plot_plotly(
        {
            "layout": layout,
            "data": [
                {
                    "type": "scatter",
                    "name": i,
                    "x": 1 + arange(errors_.size),
                    "y": errors_,
                }
                for i, errors_ in enumerate(errors)
            ],
        },
        html_file_path,
    )
