from pandas import Series

from .plot_plotly_figure import plot_plotly_figure


def plot_scatter_and_annotate(
    x,
    y,
    abs_dimension,
    annotation=(),
    opacity=0.64,
    annotation_font_size=10,
    title=None,
    html_file_path=None,
):

    if x is None:

        x = Series(range(y.size), name="Rank", index=y.index)

    if abs_dimension == "x":

        is_negative = x < 0

    elif abs_dimension == "y":

        is_negative = y < 0

    if x.size < 1e3:

        mode = "markers"

    else:

        mode = "lines"

    data = [
        {
            "type": "scatter",
            "name": "-",
            "x": x[is_negative],
            "y": y[is_negative].abs(),
            "text": y.index[is_negative],
            "mode": mode,
            "marker": {"color": "#0088ff", "opacity": opacity},
        },
        {
            "type": "scatter",
            "name": "+",
            "x": x[~is_negative],
            "y": y[~is_negative],
            "text": y.index[~is_negative],
            "mode": mode,
            "marker": {"color": "#ff1968", "opacity": opacity},
        },
    ]

    annotations = []

    for group_name, elements, size, color in annotation:

        group_elements = y.index & elements

        group_x = x[group_elements]

        group_y = y[group_elements]

        data.append(
            {
                "type": "scatter",
                "name": group_name,
                "x": group_x,
                "y": group_y.abs(),
                "text": group_elements,
                "mode": "markers",
                "marker": {
                    "size": size,
                    "color": color,
                    "line": {"width": 1, "color": "#ebf6f7"},
                },
            }
        )

        annotations += [
            {
                "x": x_,
                "y": abs(y_),
                "text": element,
                "font": {"size": annotation_font_size},
                "arrowhead": 2,
                "arrowsize": 0.8,
                "clicktoshow": "onoff",
            }
            for element, x_, y_ in zip(group_elements, group_x, group_y)
        ]

    plot_plotly_figure(
        {
            "layout": {
                "title": {"text": title},
                "xaxis": {"title": x.name},
                "yaxis": {"title": y.name},
                "annotations": annotations,
            },
            "data": data,
        },
        html_file_path,
    )
