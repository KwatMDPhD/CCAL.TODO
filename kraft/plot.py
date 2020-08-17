from numpy import arange, argsort, meshgrid, nonzero, unique
from plotly.colors import (
    convert_colors_to_same_type,
    find_intermediate_color,
    make_colorscale,
    qualitative,
)
from plotly.io import show, templates, write_html

from .array import normalize
from .dict_ import merge
from .support import cast_builtin

templates["kraft"] = {"layout": {"autosize": False}}

COLORBAR = {
    "thicknessmode": "fraction",
    "thickness": 0.024,
    "len": 0.64,
    "ticks": "outside",
    "tickfont": {"size": 10},
}

DATA_TYPE_TO_COLORSCALE = {
    "continuous": make_colorscale(("#0000ff", "#ffffff", "#ff0000")),
    "categorical": make_colorscale(qualitative.Plotly),
    "binary": make_colorscale(("#006442", "#ffffff", "#ffa400")),
}


def plot_plotly(figure, html_file_path=None):

    template = "plotly_white+kraft"

    if "layout" in figure:

        figure["layout"]["template"] = template

    else:

        figure["layout"] = {"template": template}

    config = {"editable": True}

    show(figure, config=config)

    if html_file_path is not None:

        assert html_file_path.endswith(".html")

        write_html(figure, html_file_path, config=config)


def get_color(colorscale, number, n=None):

    if n is not None:

        assert float(n).is_integer()

        assert float(number).is_integer()

        if 1 <= number < n:

            n_block = n - 1

            if number == n_block:

                number = 1.0

            else:

                number = number / n_block

    if colorscale is None:

        colorscale = make_colorscale(qualitative.Plotly)

    for i in range(len(colorscale) - 1):

        if colorscale[i][0] <= number <= colorscale[i + 1][0]:

            low_number, low_color = colorscale[i]

            high_number, high_color = colorscale[i + 1]

            color = find_intermediate_color(
                *convert_colors_to_same_type((low_color, high_color))[0],
                (number - low_number) / (high_number - low_number),
                colortype="rgb",
            )

            return "rgb{}".format(
                tuple(int(float(i)) for i in color[4:-1].split(sep=","))
            )


def plot_heat_map(
    matrix,
    colorscale=None,
    sort_groups=True,
    axis_0_groups=None,
    axis_0_group_colorscale=None,
    axis_0_group_to_name=None,
    axis_1_groups=None,
    axis_1_group_colorscale=None,
    axis_1_group_to_name=None,
    layout=None,
    layout_annotation_axis_0=None,
    layout_annotation_axis_1=None,
    html_file_path=None,
):

    if sort_groups:

        if axis_0_groups is not None:

            is_ = argsort(axis_0_groups)

            axis_0_groups = axis_0_groups[is_]

            matrix = matrix.iloc[is_, :]

        if axis_1_groups is not None:

            is_ = argsort(axis_1_groups)

            axis_1_groups = axis_1_groups[is_]

            matrix = matrix.iloc[:, is_]

    heat_map_axis_base = {"domain": (0, 0.95)}

    group_axis = {"domain": (0.96, 1), "showticklabels": False}

    axis_0_labels = matrix.index.to_numpy()

    axis_1_labels = matrix.columns.to_numpy()

    axis_0_is_str = any(isinstance(cast_builtin(label), str) for label in axis_0_labels)

    axis_1_is_str = any(isinstance(cast_builtin(label), str) for label in axis_1_labels)

    layout_base = {
        "xaxis": {
            "showticklabels": axis_1_is_str,
            "title": "{} (n={})".format(matrix.columns.name, axis_1_labels.size),
            **heat_map_axis_base,
        },
        "yaxis": {
            "showticklabels": axis_0_is_str,
            "title": "{} (n={})".format(matrix.index.name, axis_0_labels.size),
            **heat_map_axis_base,
        },
        "xaxis2": group_axis,
        "yaxis2": group_axis,
        "annotations": [],
    }

    if layout is None:

        layout = layout_base

    else:

        layout = merge(layout_base, layout)

    if axis_1_is_str:

        x = axis_1_labels

    else:

        x = None

    if axis_0_is_str:

        y = axis_0_labels[::-1]

    else:

        y = None

    if colorscale is None:

        colorscale = DATA_TYPE_TO_COLORSCALE["continuous"]

    colorbar_x = 1.05

    data = [
        {
            "type": "heatmap",
            "x": x,
            "y": y,
            "z": matrix.to_numpy()[::-1],
            "colorscale": colorscale,
            "colorbar": {**COLORBAR, "x": colorbar_x},
        }
    ]

    annotation_base = {"showarrow": False}

    if axis_0_groups is not None:

        axis_0_groups = axis_0_groups[::-1]

        colorbar_x += 0.1

        data.append(
            {
                "xaxis": "x2",
                "type": "heatmap",
                "z": axis_0_groups.reshape(axis_0_groups.size, 1),
                "colorscale": axis_0_group_colorscale,
                "colorbar": {**COLORBAR, "x": colorbar_x, "dtick": 1},
                "hoverinfo": "z+y",
            }
        )

        if axis_0_group_to_name is not None:

            layout_annotation_base = {
                "xref": "x2",
                "x": 0,
                "xanchor": "left",
                **annotation_base,
            }

            if layout_annotation_axis_0 is None:

                layout_annotation = layout_annotation_base

            else:

                layout_annotation = merge(
                    layout_annotation_base, layout_annotation_axis_0
                )

            for i in unique(axis_0_groups):

                i_0, i_1 = nonzero(axis_0_groups == i)[0][[0, -1]]

                layout["annotations"].append(
                    {
                        "y": i_0 + (i_1 - i_0) / 2,
                        "text": axis_0_group_to_name[i],
                        **layout_annotation,
                    }
                )

    if axis_1_groups is not None:

        colorbar_x += 0.1

        data.append(
            {
                "yaxis": "y2",
                "type": "heatmap",
                "z": axis_1_groups.reshape((1, axis_1_groups.size)),
                "colorscale": axis_1_group_colorscale,
                "colorbar": {**COLORBAR, "x": colorbar_x, "dtick": 1},
                "hoverinfo": "z+x",
            }
        )

        if axis_1_group_to_name is not None:

            layout_annotation_base = {
                "yref": "y2",
                "y": 0,
                "yanchor": "bottom",
                "textangle": -90,
                **annotation_base,
            }

            if layout_annotation_axis_1 is None:

                layout_annotation = layout_annotation_base

            else:

                layout_annotation = merge(
                    layout_annotation_base, layout_annotation_axis_1
                )

            for i in unique(axis_1_groups):

                i_0, i_1 = nonzero(axis_1_groups == i)[0][[0, -1]]

                layout["annotations"].append(
                    {
                        "x": i_0 + (i_1 - i_0) / 2,
                        "text": axis_1_group_to_name[i],
                        **layout_annotation,
                    }
                )

    plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)


def plot_bubble_map(
    size_matrix,
    color_matrix=None,
    max_size=32,
    colorscale=None,
    layout=None,
    html_file_path=None,
):

    axis_0_size, axis_1_size = size_matrix.shape

    x_grid = arange(axis_1_size)

    y_grid = arange(axis_0_size)[::-1]

    layout_base = {
        "height": max(480, axis_0_size * 2 * max_size),
        "width": max(480, axis_1_size * 2 * max_size),
        "xaxis": {
            "title": "{} (n={})".format(size_matrix.columns.name, axis_1_size),
            "tickvals": x_grid,
            "ticktext": size_matrix.columns.to_numpy(),
        },
        "yaxis": {
            "title": "{} (n={})".format(size_matrix.index.name, axis_0_size),
            "tickvals": y_grid,
            "ticktext": size_matrix.index.to_numpy(),
        },
    }

    if layout is None:

        layout = layout_base

    else:

        layout = merge(layout_base, layout)

    size_matrix = size_matrix.to_numpy()

    if color_matrix is None:

        color_matrix = size_matrix

    x, y = meshgrid(x_grid, y_grid)

    if colorscale is None:

        colorscale = DATA_TYPE_TO_COLORSCALE["continuous"]

    plot_plotly(
        {
            "layout": layout,
            "data": [
                {
                    "x": x.ravel(),
                    "y": y.ravel(),
                    "text": size_matrix.ravel(),
                    "mode": "markers",
                    "marker": {
                        "size": normalize(size_matrix, "0-1").ravel() * max_size,
                        "color": color_matrix.ravel(),
                        "colorscale": colorscale,
                        "colorbar": COLORBAR,
                    },
                }
            ],
        },
        html_file_path=html_file_path,
    )


def plot_histogram(
    serieses,
    histnorm=None,
    bin_size=None,
    plot_rug=None,
    layout=None,
    html_file_path=None,
):

    if plot_rug is None:

        plot_rug = all(series.size < 1e3 for series in serieses)

    n_series = len(serieses)

    if plot_rug:

        rug_height = 0.04

        yaxis_domain_max = n_series * rug_height

        yaxis2_domain_min = yaxis_domain_max + rug_height

    else:

        yaxis_domain_max = 0

        yaxis2_domain_min = 0

    yaxis_domain = 0, yaxis_domain_max

    yaxis2_domain = yaxis2_domain_min, 1

    if histnorm is None:

        yaxis2_title_text = "N"

    else:

        yaxis2_title_text = histnorm.title()

    layout_base = {
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

        layout = layout_base

    else:

        layout = merge(layout_base, layout)

    data = []

    for i, series in enumerate(serieses):

        color = get_color(DATA_TYPE_TO_COLORSCALE["categorical"], i, n_series)

        trace_base = {
            "legendgroup": i,
            "name": series.name,
            "showlegend": 1 < n_series,
            "x": series,
        }

        data.append(
            {
                "yaxis": "y2",
                "type": "histogram",
                "histnorm": histnorm,
                "xbins": {"size": bin_size},
                "marker": {"color": color},
                **trace_base,
            }
        )

        if plot_rug:

            data.append(
                {
                    "showlegend": False,
                    "y": (i,) * series.size,
                    "text": series.index.to_numpy(),
                    "mode": "markers",
                    "marker": {"symbol": "line-ns-open", "color": color},
                    "hoverinfo": "x+text",
                    **trace_base,
                }
            )

    plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)


def plot_x_y(xs, ys, xaxis_title_text, yaxis_title_text, title_text=None):

    plot_plotly(
        {
            "layout": {
                "title": {"text": title_text},
                "xaxis": {"title": {"text": xaxis_title_text}},
                "yaxis": {"title": {"text": yaxis_title_text}},
            },
            "data": [{"x": xs, "y": ys}],
        }
    )
